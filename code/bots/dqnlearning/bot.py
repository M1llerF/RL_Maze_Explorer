from __future__ import annotations

from collections import deque
from typing import Any, cast

import numpy as np

from baseBot import BaseBot
from botTools import BotTools
from bots.common.actions import ActionSpec, DEFAULT_PRIMITIVE_ACTIONS, DEFAULT_ATTACK_ACTIONS
from bots.common.actionRegistry import ActionRegistry
from bots.common.decision import ActionChoice, DecisionInput, LocalActionSpace
from bots.common.options import OptionSpec
from bots.common.optionsLibrary import defaultOptions
from bots.common.stepResult import StepResult
from environment.context import EnvironmentContext, POSITION_CHANGED
from environment.enemySystem import EnemySystem
from environment.entityFactory import build_entity_registry
from environment.entityRegistry import EntityRegistry
from environment.movement import GridMovement4Way, buildDefaultMovementExecutors, buildAttackExecutors
from environment.observationEncoder import ObservationEncoder
from environment.sensing import build_sensing_service
from environment.traversal import build_traversal_policy
from bots.botStatus import BotStatus
from botFactory import BotCreateContext
from services.episodeRecorder import PostEpisodeRecorder
from services.runners import DQNEpisodeRunner

from .agent import DqnAgent
from .checkpoint import CheckpointIO, CheckpointMeta
from .checkpointService import DQNCheckpointService
from .config import DQNConfig
from .encoder import StateEncoder
from .hrlAgent import HierarchicalDqnAgent
from .planner import WarmupPlanner
from .types import EncodedState, Transition
from .warmupCoordinator import WarmupCoordinator
from services.episodeResult import EpisodePersistencePolicy, EvaluationEpisodeDefinition


class DQNBot(BaseBot):
    config: DQNConfig  # narrows BaseBot's Any for type checkers

    def __init__(self, ctx: BotCreateContext) -> None:
        maze = ctx.maze
        config = cast(DQNConfig, ctx.config)
        rewardSystem = ctx.rewardSystem
        statistics = ctx.statistics
        profileName = ctx.profileName
        repository = ctx.repository
        loadCheckpoint = ctx.loadCheckpoint

        super().__init__(maze, statistics, config)
        self.rewardSystem = rewardSystem
        self.profileName = profileName
        self.repo = repository
        self.diagnostics = ctx.diagnostics
        self.tools = BotTools(maze, profileName)
        self._traversal = build_traversal_policy(maze, profileName)
        self._sensing = build_sensing_service(maze, profileName)
        self.position = self.maze.getStart()
        self.previousPosition: tuple[int, int] | None = None
        self.lastAction: int | None = None
        self.totalReward = 0.0
        self.episodeCounter = 0
        self._knownOpen: set[tuple[int, int]] = set()
        self._knownWalls: set[tuple[int, int]] = set()
        self._seenGoals: set[tuple[int, int]] = set()
        self._collectingWarmup = False
        # Persistent numpy arrays for fast map encoding.
        # _knownMap channels: [knownOpen, knownWall, seenGoal]
        # _visitedMap: has this cell been visited this episode?
        self._knownMap = np.zeros((1, 1, 3), dtype=np.float32)
        self._visitedMap = np.zeros((1, 1), dtype=np.float32)
        self._entityRegistry = EntityRegistry()
        self._movementModel = GridMovement4Way()
        self._actionRegistry = ActionRegistry()
        self._optionSpecs: list[OptionSpec] = []
        self._optionSelectionCounts: dict[str, int] = {}
        self._optionDurationTotals: dict[str, int] = {}
        self._optionSuccessCounts: dict[str, int] = {}
        executors = buildDefaultMovementExecutors(self._movementModel)
        for spec in DEFAULT_PRIMITIVE_ACTIONS:
            self._actionRegistry.registerPrimitive(spec, executors[spec.id])
        if getattr(self.config, "useAttackActions", False):
            attack_executors = buildAttackExecutors()
            for spec in DEFAULT_ATTACK_ACTIONS:
                self._actionRegistry.registerPrimitive(spec, attack_executors[spec.id])
        if getattr(self.config, "useMacroActions", False):
            optionSet = str(getattr(self.config, "macroOptionSet", "naive"))
            for option in defaultOptions(optionSet):
                self._optionSpecs.append(option)
                self._actionRegistry.registerOption(option)
        self._context = EnvironmentContext(
            maze=self.maze,
            bot=self,
            entity_registry=self._entityRegistry,
            action_registry=self._actionRegistry,
            traversal_policy=self._traversal,
        )
        self._context.on(POSITION_CHANGED, self._onPositionChanged)
        self._observationEncoder = ObservationEncoder(
            scale=float(max(int(self.maze.height), int(self.maze.width), 1))
        )
        self._syncMazeDependentState()
        self._syncEntitiesFromMaze()
        self._observePosition(cast(tuple[int, int], self.position))
        self.state = self.calculateState()
        schema = self.encoder.inferSchema(self.state)
        self._sequenceFrameDim = int(schema.inputDim)
        self._encodedFrameHistory: deque[np.ndarray] = deque(maxlen=self._sequenceLength())
        self._rebuildEncodedFrameHistory()
        policyInputDim = self._policyInputDim(self._sequenceFrameDim)
        self._policyMode = self._resolvePolicyMode()
        flatActionCount = self._flatActionSpace().numActions
        optionActionCount = self._optionActionSpace().numActions
        if self._policyActionCount() <= 0:
            raise ValueError(
                "DQNBot resolved to an empty policy action space. "
                "Enable macro actions for macro-only/hierarchical modes, or disable macro-only mode."
            )
        self.agent: HierarchicalDqnAgent | DqnAgent
        self.lowAgent: DqnAgent | None
        if self.usesHierarchicalPolicy:
            self.agent = HierarchicalDqnAgent(
                self.config,
                inputDim=policyInputDim,
                optionCount=optionActionCount,
                primitiveCount=self._primitiveActionSpace().numActions,
                flatDim=schema.flatDim,
                mapShape=schema.mapShape,
            )
            self.lowAgent = self.agent.lowLevel
        else:
            self.agent = DqnAgent(
                self.config,
                inputDim=policyInputDim,
                numActions=flatActionCount,
                flatDim=schema.flatDim,
                mapShape=schema.mapShape,
            )
            self.lowAgent = None
        self.primitiveBootstrapAgent = None
        if self.dualRecordsPrimitivePolicy:
            self.primitiveBootstrapAgent = DqnAgent(
                self.config,
                inputDim=policyInputDim,
                numActions=self._primitiveActionSpace().numActions,
                flatDim=schema.flatDim,
                mapShape=schema.mapShape,
            )
        self.checkpoint = CheckpointIO(self.repo, self.profileName, self.agent.device)
        self.primitiveCheckpoint = CheckpointIO(
            self.repo,
            self.profileName,
            self.agent.device,
            artifactName="dqn_model_primitive.pt",
        )
        self.legacyPrimitiveCheckpoint = CheckpointIO(
            self.repo,
            self.profileName,
            self.agent.device,
            artifactName="dqn_model_primtive.pt",
        )
        self.checkpointMeta = CheckpointMeta(
            stateSchemaVersion=schema.stateSchemaVersion,
            encoderConfigFingerprint=schema.encoderConfigFingerprint,
            inputDim=policyInputDim,
            actionDim=self._policyActionCount(),
            policyMode=self._policyMode,
        )
        self.primitiveCheckpointMeta = CheckpointMeta(
            stateSchemaVersion=schema.stateSchemaVersion,
            encoderConfigFingerprint=schema.encoderConfigFingerprint,
            inputDim=policyInputDim,
            actionDim=self._primitiveActionSpace().numActions,
            policyMode="flat",
        )
        self._planner = WarmupPlanner(
            maze=self.maze,
            profileName=self.profileName,
            tools=self.tools,
            knownOpen=self._knownOpen,
            knownWalls=self._knownWalls,
            seenGoals=self._seenGoals,
        )
        self.warmupCoordinator = WarmupCoordinator(self._planner)
        self.checkpointService = DQNCheckpointService(
            self.checkpoint,
            self.primitiveCheckpoint,
            self.checkpointMeta,
            self.primitiveCheckpointMeta,
            profileName=self.profileName,
            legacyPrimitiveCheckpoint=self.legacyPrimitiveCheckpoint,
            diagnostics=self.diagnostics,
        )
        self.runner = DQNEpisodeRunner(self)
        self.recorder = PostEpisodeRecorder(self.repo)
        self._enemySystem = EnemySystem()
        self._visInitialized = False
        self._visActiveOption: OptionSpec | None = None
        self._visActiveOptionLocalId: int | None = None
        self._visActiveOptionDuration: int = 0
        self._visDebugEnabled = False
        self._visDebugStep = 0
        self._visRecentPositions: list[tuple[int, int]] = []
        self._recentPositions: list[tuple[int, int]] = [cast(tuple[int, int], self.position)]
        self._visWallHits: int = 0
        self._visEnemyKills: int = 0
        self._visStepCount: int = 0
        self._visOutcome: str = ""
        if loadCheckpoint:
            self._loadCheckpoint()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._syncMazeDependentState()
        self._syncEntitiesFromMaze()
        self.position = self.maze.getStart()
        self.previousPosition = None
        self.lastAction = None
        # Clear numpy maps before calculateState so the initial observation is clean.
        self._knownMap[:] = 0.0
        self._visitedMap[:] = 0.0
        self.statistics.reset()
        start = cast(tuple[int, int], self.position)
        self.statistics.updateVisitedPositions(start)
        self._visitedMap[start[0], start[1]] = 1.0
        self.totalReward = 0.0
        self._knownOpen.clear()
        self._knownWalls.clear()
        self._seenGoals.clear()
        self._recentPositions = [start]
        self._pushCooldownRemaining = 0
        self.warmupCoordinator.reset()
        self._observePosition(cast(tuple[int, int], self.position))
        self.state = self.calculateState()
        self._rebuildEncodedFrameHistory()

    def _syncMazeDependentState(self) -> None:
        mazeH = max(1, int(self.maze.height))
        mazeW = max(1, int(self.maze.width))
        if self._knownMap.shape != (mazeH, mazeW, 3):
            self._knownMap = np.zeros((mazeH, mazeW, 3), dtype=np.float32)
        if self._visitedMap.shape != (mazeH, mazeW):
            self._visitedMap = np.zeros((mazeH, mazeW), dtype=np.float32)
        self._observationEncoder = ObservationEncoder(scale=float(max(mazeH, mazeW, 1)))
        self.encoder = StateEncoder(self.config, mazeHeight=mazeH, mazeWidth=mazeW)

    # ------------------------------------------------------------------
    # Episode hooks (BaseBot interface)
    # ------------------------------------------------------------------

    def runEpisode(self) -> None:
        result = self.runner.runEpisode()
        self.recorder.record(self, result)

    def runEvaluationEpisode(self) -> None:
        result = self.runner.runEpisode(mode="evaluation")
        self.recorder.record(self, result)

    def applyLearningUpdate(
        self,
        encodedState: Any,
        localId: int,
        reward: float,
        done: bool,
        duration: int = 1,
    ) -> None:
        """Store transition and run a training step. Called by DQNEpisodeRunner."""
        nextDecision = self.buildDecisionInput(training=True)
        nextEncoded = nextDecision.state
        transition = self.makeTransition(
            cast(EncodedState, encodedState), localId, reward,
            cast(EncodedState, nextEncoded), done, duration=duration,
        )
        cast(Any, self.agent).storeTransition(transition)
        self.agent.trainStep()
        self.agent.onEnvironmentStep()

    def applyHighLevelLearningUpdate(
        self,
        encodedState: Any,
        localId: int,
        reward: float,
        done: bool,
        duration: int = 1,
    ) -> None:
        """Store high-level transition and run a training step for hierarchical policy."""
        nextDecision = self.buildHighLevelDecisionInput()
        nextEncoded = nextDecision.state
        transition = self.makeTransition(
            cast(EncodedState, encodedState), localId, reward,
            cast(EncodedState, nextEncoded), done, duration=duration,
        )
        cast(Any, self.agent).storeHighTransition(transition)
        self.agent.trainStep()

    def persistTrainingArtifacts(self) -> None:
        self.saveCheckpoint()

    def getTrainingEpisodePersistencePolicy(self) -> EpisodePersistencePolicy:
        return EpisodePersistencePolicy(
            save_maze_episode=True,
            save_heatmap_stats=True,
            append_reward=True,
        )

    def getEvaluationEpisodeDefinition(self) -> EvaluationEpisodeDefinition:
        return EvaluationEpisodeDefinition(
            description=(
                "Greedy evaluation episode with learning disabled and epsilon forced to 0.0. "
                "Evaluation snapshots latest/highest/lowest mazes and reward history."
            ),
            frequency=int(getattr(self.config, "evaluationFrequency", 0)),
            is_dedicated_episode=True,
            eval_epsilon=0.0,
            persistence=EpisodePersistencePolicy(
                save_maze_episode=True,
                save_heatmap_stats=False,
                append_reward=True,
            ),
        )

    # ── BaseBot public contract implementations ───────────────────────────────

    def getStatus(self) -> BotStatus:
        replaySize = len(cast(Any, self.agent).replay)
        warmupRequired = int(getattr(self.config, "replayWarmupSteps", 0))
        explorationRate = 0.0
        epsilonIsManual = False
        if hasattr(self.agent, "diagnostics"):
            diag = cast(Any, self.agent).diagnostics()
            explorationRate = float(diag.epsilon)
            epsilonIsManual = getattr(cast(Any, self.agent), "manualEpsilonOverride", None) is not None
        return BotStatus(
            episode_count=int(self.episodeCounter),
            is_warming_up=self.isWarmingUp,
            replay_size=replaySize,
            warmup_required=warmupRequired,
            current_exploration_rate=explorationRate,
            epsilon_is_manual=epsilonIsManual,
            checkpoint_frequency=int(getattr(self.config, "checkpointFrequency", 0)),
            current_episode_steps=int(getattr(self, "currentEpisodeSteps", 0)),
            last_episode_success=getattr(self, "lastEpisodeSuccess", False),
            artifact_save_label="checkpoint",
        )

    def optionDiagnostics(self) -> dict[str, object]:
        names = sorted(set(self._optionSelectionCounts) | set(self._optionDurationTotals) | set(self._optionSuccessCounts))
        per_option: dict[str, dict[str, float | int]] = {}
        for name in names:
            count = int(self._optionSelectionCounts.get(name, 0))
            duration = int(self._optionDurationTotals.get(name, 0))
            successes = int(self._optionSuccessCounts.get(name, 0))
            per_option[name] = {
                "selection_count": count,
                "total_duration": duration,
                "mean_duration": round(duration / count, 4) if count else 0.0,
                "success_count": successes,
            }
        return {
            "macro_option_set": str(getattr(self.config, "macroOptionSet", "naive")),
            "registered_options": [option.name for option in self._optionSpecs],
            "total_option_selections": sum(int(v) for v in self._optionSelectionCounts.values()),
            "per_option": per_option,
        }

    def setManualEpsilon(self, value: float | None) -> bool:
        if not hasattr(self.agent, "setManualEpsilon"):
            return False
        cast(Any, self.agent).setManualEpsilon(value)
        return True

    def onEpisodeStart(self, mode: str) -> None:
        if mode not in {"training", "evaluation"}:
            raise ValueError(f"Unsupported episode mode: {mode}")

    def onEpisodeStep(self, mode: str, stepIndex: int) -> None:
        if mode not in {"training", "evaluation"}:
            raise ValueError(f"Unsupported episode mode: {mode}")
        if stepIndex < 0:
            raise ValueError("stepIndex must be non-negative")

    def onEpisodeEnd(self, mode: str, outcome: str) -> None:
        if mode not in {"training", "evaluation"}:
            raise ValueError(f"Unsupported episode mode: {mode}")
        if not outcome:
            raise ValueError("outcome must be non-empty")

    # ------------------------------------------------------------------
    # Public step interface (used by DQNEpisodeRunner)
    # ------------------------------------------------------------------

    @property
    def isWarmingUp(self) -> bool:
        return self._collectingWarmup

    @property
    def usesHierarchicalPolicy(self) -> bool:
        return getattr(self.config, "useHierarchicalPolicy", False)

    @property
    def usesMacroOnlyPolicy(self) -> bool:
        return getattr(self.config, "useMacroOnlyPolicy", False)

    @property
    def dualRecordsPrimitivePolicy(self) -> bool:
        return getattr(self.config, "dualRecordPrimitivePolicy", False)

    def applyStep(self, action_id: int, training: bool = False) -> StepResult:
        """Apply a local action chosen from the active policy action space."""
        if self.usesHierarchicalPolicy:
            return self.executeHierarchicalOption(action_id, training=training)

        actionSpace = self._defaultPolicyActionSpace()
        if action_id < 0 or action_id >= actionSpace.numActions:
            return StepResult(reward=0.0, done=False, info={"hit_wall": True}, duration=1)
        semanticAction = self._actionRegistry.semanticIdForLocal(actionSpace, action_id)
        self.lastAction = int(semanticAction)
        if self._actionRegistry.isOption(semanticAction):
            result = self._applyFlatOption(semanticAction, training=training)
        else:
            result = self._applyPrimitiveStep(semanticAction)
        return result

    # ── Shared action contract (BaseBot interface) ────────────────────────────

    def getActionSpace(self) -> list[ActionSpec]:
        return self._actionRegistry.getActionSpace()

    def getPolicyActionSpace(self) -> list[ActionSpec]:
        return self._defaultPolicyActionSpace().actionList()

    def getValidActionMask(self) -> list[bool]:
        return self._defaultPolicyActionSpace().maskList()

    def buildDecisionInput(self, *, training: bool) -> DecisionInput:
        if self.usesHierarchicalPolicy:
            return self.buildHighLevelDecisionInput()
        actionSpace = self._defaultPolicyActionSpace()
        encoded = self.encodeState(actionSpace=actionSpace)
        return DecisionInput(state=encoded, actionSpace=actionSpace, level="flat")

    def buildHighLevelDecisionInput(self) -> DecisionInput:
        actionSpace = self._optionActionSpace()
        encoded = self.encodeState(actionSpace=actionSpace)
        return DecisionInput(state=encoded, actionSpace=actionSpace, level="high")

    def buildLowLevelDecisionInput(self, optionLocalId: int) -> DecisionInput:
        optionSpace = self._optionActionSpace()
        primitiveSpace = self._primitiveActionSpace()
        baseEncoded = self.encodeState(actionSpace=optionSpace)
        optionVector = np.zeros(optionSpace.numActions, dtype=np.float32)
        if 0 <= int(optionLocalId) < optionSpace.numActions:
            optionVector[int(optionLocalId)] = 1.0
        augmented = EncodedState(
            values=np.concatenate([baseEncoded.values, optionVector]).astype(np.float32),
            validActionMask=np.asarray(primitiveSpace.validActionMask, dtype=np.bool_),
        )
        return DecisionInput(
            state=augmented,
            actionSpace=primitiveSpace,
            level="low",
            activeOptionLocalId=int(optionLocalId),
        )

    def addReward(self, amount: float, reason: str = "") -> None:
        self.totalReward += amount

    def shapingPenalty(self, wasRevisit: bool = False, wasReversal: bool = False) -> float:
        """Config-driven per-step shaping penalties, consolidated in one place."""
        penalty = 0.0
        if wasRevisit:
            penalty += float(self.config.repeatVisitPenaltyScale)
        if wasReversal:
            penalty += float(self.config.immediateReversalPenalty)
        return penalty

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _observationScale(self) -> float:
        return float(max(
            1,
            int(getattr(self.config, "neuralMapHeight", 31)),
            int(getattr(self.config, "neuralMapWidth", 31)),
        ))

    def _sequenceLength(self) -> int:
        return max(1, int(getattr(self.config, "lstmSequenceLength", 1)))

    def _policyInputDim(self, baseDim: int) -> int:
        return int(baseDim) * self._sequenceLength() if getattr(self.config, "useLstmPolicy", False) else int(baseDim)

    def _encodeBaseState(self, observation: tuple[Any, ...]) -> EncodedState:
        return self.encoder.encode(observation)

    def _rebuildEncodedFrameHistory(self) -> None:
        self._encodedFrameHistory.clear()
        if not getattr(self.config, "useLstmPolicy", False):
            return
        self._encodedFrameHistory.append(self._encodeBaseState(self.state).values.copy())

    def _appendCurrentEncodedFrame(self) -> None:
        if not getattr(self.config, "useLstmPolicy", False):
            return
        self._encodedFrameHistory.append(self._encodeBaseState(self.state).values.copy())

    def _stackSequenceValues(self, currentValues: np.ndarray) -> np.ndarray:
        if not getattr(self.config, "useLstmPolicy", False):
            return np.asarray(currentValues, dtype=np.float32)
        zeroFrame = np.zeros(int(self._sequenceFrameDim), dtype=np.float32)
        frames = [np.asarray(frame, dtype=np.float32) for frame in self._encodedFrameHistory]
        if not frames:
            frames = [np.asarray(currentValues, dtype=np.float32)]
        elif not np.array_equal(frames[-1], currentValues):
            frames = frames[:-1] + [np.asarray(currentValues, dtype=np.float32)]
        frames = frames[-self._sequenceLength():]
        if len(frames) < self._sequenceLength():
            frames = [zeroFrame.copy() for _ in range(self._sequenceLength() - len(frames))] + frames
        return np.concatenate(frames, axis=0).astype(np.float32, copy=False)

    def calculateState(self, actionSpace: LocalActionSpace | None = None) -> tuple[Any, ...]:
        currentActionSpace = actionSpace or self._defaultPolicyActionSpace()
        if getattr(self.config, "useSharedComparisonState", False):
            compactState = self._observationEncoder.encodeCompactState(
                self._context,
                self._sensing,
                position=cast(tuple[int, int], self.position),
                include_entity_features=getattr(self.config, "useEntityObservation", False),
                include_enemy_features=getattr(self.config, "useEnemyObservation", False),
            )
            validActions = tuple(1 if v else 0 for v in currentActionSpace.validActionMask)
            return compactState + (validActions,)

        position = cast(tuple[int, int], self.position)
        wallDistances, _ = self._sensing.wall_distances_and_goal_directions(position)
        previousDelta = (
            (0, 0)
            if self.previousPosition is None
            else (self.previousPosition[0] - position[0], self.previousPosition[1] - position[1])
        )
        validActions = tuple(1 if v else 0 for v in currentActionSpace.validActionMask)
        # Local observation: first-person line-of-sight rays per direction.
        localObservation = self._calculateLocalObservation(position)
        neuralMap: tuple[float, ...] | np.ndarray = ()
        if self.config.useRichEncoding:
            neuralMap = self.encoder.encodeNeuralMap(position, self._knownMap, self._visitedMap)
        entityFeatures: tuple[float, ...] = ()
        if getattr(self.config, "useEntityObservation", False):
            entityFeatures = tuple(float(v) for v in self._observationEncoder.encode(self._context).tolist())
        if getattr(self.config, "useEnemyObservation", False):
            enemy_scan = self._observationEncoder.encodeEnemyScan(self._context, self._sensing)
            entityFeatures = entityFeatures + enemy_scan
        return (
            position,
            wallDistances,
            previousDelta,
            -1 if self.lastAction is None else self.lastAction,
            localObservation,
            neuralMap,
            validActions,
            entityFeatures,
        )

    def encodeState(
        self,
        state: tuple[Any, ...] | None = None,
        *,
        actionSpace: LocalActionSpace | None = None,
    ) -> EncodedState:
        observation = self.calculateState(actionSpace=actionSpace) if state is None else state
        baseEncoded = self._encodeBaseState(observation)
        if not getattr(self.config, "useLstmPolicy", False):
            return baseEncoded
        return EncodedState(
            values=self._stackSequenceValues(baseEncoded.values),
            validActionMask=baseEncoded.validActionMask,
        )

    def chooseAction(
        self,
        training: bool,
        *,
        forceEpsilon: float | None = None,
        allowWarmupPlanner: bool = True,
    ) -> ActionChoice:
        decision = self.buildDecisionInput(training=training)
        autoAttackLocalId = self.autoAttackAdjacentEnemyLocalId(decision.actionSpace)
        if autoAttackLocalId is not None:
            return ActionChoice(local_id=autoAttackLocalId)
        if isinstance(self.agent, DqnAgent):
            warmupAction = self.warmupCoordinator.selectAction(
                self,
                decision,
                training=training,
                allowWarmupPlanner=allowWarmupPlanner,
            )
            if warmupAction is not None:
                return warmupAction
            if forceEpsilon is not None:
                self.agent.setManualEpsilon(float(forceEpsilon))
            try:
                return self.agent.chooseAction(decision, training=bool(training or forceEpsilon is not None))
            finally:
                if forceEpsilon is not None:
                    self.agent.setManualEpsilon(None)
        if forceEpsilon is not None:
            self.agent.setManualEpsilon(float(forceEpsilon))
        try:
            return self.agent.chooseHighLevelAction(
                decision,
                training=bool(training or forceEpsilon is not None),
            )
        finally:
            if forceEpsilon is not None:
                self.agent.setManualEpsilon(None)

    def selectWarmupAction(self, decision: Any, plannerLocal: int | None) -> ActionChoice:
        return self.agent.selectWarmupAction(decision, plannerLocal)

    def plannerDirectionToOptionLocalId(self, decision: DecisionInput, primitiveDirection: int) -> int | None:
        """Map a planner primitive direction to the local ID of the matching option.

        Iterates valid options in the decision's action space and returns the
        local ID of the first option whose initial primitive step matches the
        requested direction. Returns None if no valid option covers that direction.
        """
        for localId in range(decision.actionSpace.numActions):
            if not decision.actionSpace.validActionMask[localId]:
                continue
            semanticId = decision.actionSpace.semanticId(localId)
            option = self._actionRegistry.getOption(semanticId)
            if option is None:
                continue
            if option.choosePrimitiveAction(self._context) == primitiveDirection:
                return localId
        return None

    def makeTransition(
        self,
        encodedState: EncodedState,
        action: int,
        reward: float,
        nextEncodedState: EncodedState,
        done: bool,
        duration: int = 1,
    ) -> Transition:
        return Transition(
            state=encodedState.values,
            action=action,
            reward=reward,
            nextState=nextEncodedState.values,
            done=done,
            validActionMask=encodedState.validActionMask,
            nextValidActionMask=nextEncodedState.validActionMask,
            bootstrapDiscount=float(self.config.discountFactor) ** max(1, int(duration)),
        )

    def executeHierarchicalOption(self, optionLocalId: int, *, training: bool) -> StepResult:
        if not isinstance(self.agent, HierarchicalDqnAgent):
            raise RuntimeError("Hierarchical option execution is only available in hierarchical mode")
        optionSpace = self._optionActionSpace()
        semanticOptionId = optionSpace.semanticId(int(optionLocalId))
        option = self._actionRegistry.getOption(semanticOptionId)
        if option is None:
            raise KeyError(f"Unknown option action_id={semanticOptionId}")

        cumulativeReward = 0.0
        discountedReward = 0.0
        duration = 0
        revisitCount = 0
        reversalCount = 0
        hitWallCount = 0
        done = False
        while not option.shouldTerminate(self._context, duration) and not done:
            lowDecision = self.buildLowLevelDecisionInput(optionLocalId)
            primitiveChoice = self.agent.chooseLowLevelAction(lowDecision, training=training)
            primitiveSemantic = lowDecision.actionSpace.semanticId(primitiveChoice.local_id)
            primitiveResult = self._applyPrimitiveStep(primitiveSemantic)
            optionReward = float(primitiveResult.reward) + float(option.intrinsicReward(self._context, primitiveResult))
            cumulativeReward += optionReward
            discountedReward += (float(self.config.discountFactor) ** duration) * optionReward
            duration += int(primitiveResult.duration)
            revisitCount += 1 if bool(primitiveResult.info.get("was_revisit", False)) else 0
            reversalCount += 1 if bool(primitiveResult.info.get("was_reversal", False)) else 0
            hitWallCount += 1 if bool(primitiveResult.info.get("hit_wall", False)) else 0
            done = bool(primitiveResult.done or self.position == self.maze.end)

            nextLowDecision = self.buildLowLevelDecisionInput(optionLocalId)
            lowTransitionDone = bool(done or option.shouldTerminate(self._context, duration))
            if training:
                lowTransition = Transition(
                    state=cast(EncodedState, lowDecision.state).values,
                    action=primitiveChoice.local_id,
                    reward=optionReward,
                    nextState=cast(EncodedState, nextLowDecision.state).values,
                    done=lowTransitionDone,
                    validActionMask=cast(EncodedState, lowDecision.state).validActionMask,
                    nextValidActionMask=cast(EncodedState, nextLowDecision.state).validActionMask,
                    bootstrapDiscount=float(self.config.discountFactor),
                )
                self.agent.storeLowTransition(lowTransition)
                self.agent.trainStep()
                self.agent.onEnvironmentStep()
            if done:
                break

        self.state = self.calculateState()
        return StepResult(
            reward=cumulativeReward,
            done=done,
            info={
                "hit_wall": hitWallCount > 0,
                "was_revisit": revisitCount > 0,
                "was_reversal": reversalCount > 0,
                "option": option.name,
                "action_key": option.name,
            },
            duration=max(1, duration),
            trainingReward=discountedReward,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def saveCheckpoint(self) -> None:
        self.checkpointService.save(self)
        # The replay buffer is not persisted — serialising 15 k × 1600-float
        # transitions to disk on every save blocks the training thread for several
        # seconds and is the primary cause of the "steps frozen" symptom.
        # Model weights are small (~100 KB) and save in < 1 ms.

    def _loadCheckpoint(self) -> None:
        self.checkpointService.load(self)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def beginVisualizationEpisode(self) -> None:
        self.reset()
        self._visInitialized = True
        self._visActiveOption = None
        self._visActiveOptionLocalId = None
        self._visActiveOptionDuration = 0
        self._visDebugStep = 0
        self._visRecentPositions = [cast(tuple[int, int], self.position)]
        self._visWallHits = 0
        self._visEnemyKills = 0
        self._visStepCount = 0
        self._visOutcome = ""
        if self._visDebugEnabled:
            self._log_visualization_debug(
                "Visualization episode began",
                policy=self._policyMode,
                position=self.position,
            )

    def stepVisualization(self, maxSteps: int = 1) -> bool:
        if not self._visInitialized:
            self.beginVisualizationEpisode()
        stepLimit = int(getattr(self.config, "maxStepsPerEpisode", 3 * int(self.maze.width) * int(self.maze.height)))
        for _ in range(maxSteps):
            if self.position == self.maze.end:
                self._visOutcome = "goal_reached"
                self._visInitialized = False
                self._visActiveOption = None
                self._visActiveOptionLocalId = None
                self._visActiveOptionDuration = 0
                return True
            self._stepVisualizationOnce()
            tick = self._enemySystem.tick(self._context)
            self._visEnemyKills += len(tick.defeated_enemy_ids)
            self._visStepCount += 1
            if tick.agent_caught:
                self._visOutcome = "death_by_enemy"
                self._visInitialized = False
                self._visActiveOption = None
                self._visActiveOptionLocalId = None
                self._visActiveOptionDuration = 0
                return True
            if self._visStepCount >= stepLimit:
                self._visOutcome = "step_limit"
                self._visInitialized = False
                self._visActiveOption = None
                self._visActiveOptionLocalId = None
                self._visActiveOptionDuration = 0
                return True
            if self.position == self.maze.end:
                self._visOutcome = "goal_reached"
                self._visInitialized = False
                self._visActiveOption = None
                self._visActiveOptionLocalId = None
                self._visActiveOptionDuration = 0
                return True
        return False

    def _stepVisualizationOnce(self) -> None:
        """Execute exactly one primitive step for visualization, respecting active options."""
        if self._visActiveOption is not None:
            option = self._visActiveOption
            if option.shouldTerminate(self._context, self._visActiveOptionDuration) or self.position == self.maze.end:
                if self._visDebugEnabled:
                    self._log_visualization_debug(
                        "Visualization option ended",
                        step=int(self._visDebugStep),
                        option_name=option.name,
                        duration=int(self._visActiveOptionDuration),
                        position=self.position,
                    )
                self._visActiveOption = None
                self._visActiveOptionLocalId = None
                self._visActiveOptionDuration = 0
                self.state = self.calculateState()
                return
            if self.usesHierarchicalPolicy and self._visActiveOptionLocalId is not None:
                lowDecision = self.buildLowLevelDecisionInput(self._visActiveOptionLocalId)
                primitiveChoice = cast(HierarchicalDqnAgent, self.agent).chooseLowLevelAction(lowDecision, training=False)
                primitiveSemantic = lowDecision.actionSpace.semanticId(int(primitiveChoice.local_id))
            else:
                primitiveSemantic = option.choosePrimitiveAction(self._context)
            startPos = cast(tuple[int, int], self.position)
            result = self._applyPrimitiveStep(primitiveSemantic)
            if result.info.get("hit_wall", False):
                self._visWallHits += 1
            self._visDebugStep += 1
            self._visActiveOptionDuration += 1
            if self._visDebugEnabled:
                self._visLogPrimitiveStep(
                    startPos,
                    cast(tuple[int, int], self.position),
                    primitiveSemantic,
                    source=f"option:{option.name}",
                    duration=self._visActiveOptionDuration,
                    result=result,
                )
            if option.shouldTerminate(self._context, self._visActiveOptionDuration) or self.position == self.maze.end:
                if self._visDebugEnabled:
                    self._log_visualization_debug(
                        "Visualization option ended",
                        step=int(self._visDebugStep),
                        option_name=option.name,
                        duration=int(self._visActiveOptionDuration),
                        position=self.position,
                    )
                self._visActiveOption = None
                self._visActiveOptionLocalId = None
                self._visActiveOptionDuration = 0
            self.state = self.calculateState()
            return

        choice = self.chooseAction(training=False)
        actionSpace = self._defaultPolicyActionSpace()
        if choice.local_id < 0 or choice.local_id >= actionSpace.numActions:
            self.state = self.calculateState()
            return
        semanticAction = self._actionRegistry.semanticIdForLocal(actionSpace, choice.local_id)
        self.lastAction = int(semanticAction)
        if self._actionRegistry.isOption(semanticAction):
            option = self._actionRegistry.getOption(semanticAction)
            if option is not None:
                if self._visDebugEnabled:
                    self._log_visualization_debug(
                        "Visualization option chosen",
                        step=int(self._visDebugStep),
                        local_id=int(choice.local_id),
                        semantic_action=int(semanticAction),
                        option_name=option.name,
                        position=self.position,
                    )
                self._visActiveOption = option
                self._visActiveOptionLocalId = choice.local_id if self.usesHierarchicalPolicy else None
                self._visActiveOptionDuration = 0
                self._stepVisualizationOnce()
        else:
            startPos = cast(tuple[int, int], self.position)
            result = self._applyPrimitiveStep(semanticAction)
            if result.info.get("hit_wall", False):
                self._visWallHits += 1
            self._visDebugStep += 1
            if self._visDebugEnabled:
                self._visLogPrimitiveStep(
                    startPos,
                    cast(tuple[int, int], self.position),
                    semanticAction,
                    source="direct_primitive",
                    duration=1,
                    result=result,
                )
            self.state = self.calculateState()

    def setVisualizationDebug(self, enabled: bool) -> None:
        self._visDebugEnabled = bool(enabled)
        self._visDebugStep = 0
        self._visRecentPositions = [cast(tuple[int, int], self.position)]

    def _visActionName(self, semanticAction: int) -> str:
        for spec in self._actionRegistry.getActionSpace():
            if int(spec.id) == int(semanticAction):
                return str(spec.name)
        return f"action_{int(semanticAction)}"

    def _visLogPrimitiveStep(
        self,
        startPos: tuple[int, int],
        endPos: tuple[int, int],
        primitiveSemantic: int,
        *,
        source: str,
        duration: int,
        result: StepResult,
    ) -> None:
        actionName = self._visActionName(primitiveSemantic)
        hitWall = bool(result.info.get("hit_wall", False))
        wasRevisit = bool(result.info.get("was_revisit", False))
        wasReversal = bool(result.info.get("was_reversal", False))
        self._log_visualization_debug(
            "Visualization primitive step",
            step=int(self._visDebugStep),
            source=source,
            action_name=actionName,
            semantic_action=int(primitiveSemantic),
            start_position=startPos,
            end_position=endPos,
            hit_wall=hitWall,
            revisit=wasRevisit,
            reversal=wasReversal,
            option_step=int(duration),
            reward=float(result.reward),
        )
        self._visRecentPositions.append(endPos)
        if len(self._visRecentPositions) > 4:
            self._visRecentPositions.pop(0)
        if (
            len(self._visRecentPositions) == 4
            and self._visRecentPositions[0] == self._visRecentPositions[2]
            and self._visRecentPositions[1] == self._visRecentPositions[3]
            and self._visRecentPositions[0] != self._visRecentPositions[1]
        ):
            self._log_visualization_debug(
                "Visualization oscillation detected",
                pattern=list(self._visRecentPositions),
                active_option=getattr(self._visActiveOption, "name", None),
            )

    def _log_visualization_debug(self, message: str, **context: Any) -> None:
        if self.diagnostics is not None:
            self.diagnostics.debug(
                "dqn_visualization",
                message,
                profile_name=self.profileName,
                **context,
            )

    # ------------------------------------------------------------------
    # Observation helpers (private — called only from within this class)
    # ------------------------------------------------------------------

    def _observePosition(self, position: tuple[int, int]) -> None:
        goal = cast(tuple[int, int], self.maze.end)
        visible_open = self._sensing.visible_open_cells(position)
        blocked_cells = self._sensing.first_blocked_cells(position)

        for row, col in visible_open:
            self._knownOpen.add((row, col))
            self._knownWalls.discard((row, col))
            self._knownMap[row, col, 0] = 1.0
            self._knownMap[row, col, 1] = 0.0
            self._visitedMap[row, col] = 1.0 if (row, col) == position else self._visitedMap[row, col]
            if (row, col) == goal:
                self._seenGoals.add((row, col))
                self._knownMap[row, col, 2] = 1.0

        self._visitedMap[position[0], position[1]] = 1.0
        for row, col in blocked_cells:
            if (row, col) not in self._knownOpen:
                self._knownWalls.add((row, col))
                self._knownMap[row, col, 1] = 1.0

    def _onPositionChanged(self, *, new_position: tuple[int, int], **_: Any) -> None:
        self._observePosition(new_position)

    def _rememberWall(self, position: tuple[int, int]) -> None:
        if position not in self._knownOpen:
            self._knownWalls.add(position)
            if 0 <= position[0] < int(self.maze.height) and 0 <= position[1] < int(self.maze.width):
                self._knownMap[position[0], position[1], 1] = 1.0
        self.warmupCoordinator.onWallDiscovered()

    def _recordPositionAndDetectOscillation(self, position: tuple[int, int]) -> bool:
        self._recentPositions.append(position)
        if len(self._recentPositions) > 4:
            self._recentPositions.pop(0)
        if len(self._recentPositions) < 4:
            return False
        p0, p1, p2, p3 = self._recentPositions
        return p0 == p2 and p1 == p3 and p0 != p1

    def _applyPrimitiveStep(self, action: int) -> StepResult:
        self.lastAction = int(action)
        currentPos = cast(tuple[int, int], self.position)
        attempted = self._movementModel.candidatePosition(currentPos, action)
        wasReversal = self.previousPosition is not None and attempted == self.previousPosition
        wasRevisit = attempted in self.statistics.getVisitedPositions()
        rewardContext = self.rewardSystem.buildStepContext(
            currentPos,
            attempted,
            cast(list[tuple[int, int]], getattr(self, "_currentOptimalPath", [])),
            int(getattr(self, "_currentOptimalLength", 0)),
            self.statistics.getVisitedPositions(),
        )
        reward = float(
            self.rewardSystem.evaluateStep(rewardContext)
            if hasattr(self.rewardSystem, "evaluateStep")
            else self.rewardSystem.getReward(
                currentPos,
                attempted,
                cast(list[tuple[int, int]], getattr(self, "_currentOptimalPath", [])),
                int(getattr(self, "_currentOptimalLength", 0)),
                self.statistics.getVisitedPositions(),
            )
        )
        result = self._actionRegistry.executePrimitive(action, self._context)
        hitWall = bool(result.info.get("hit_wall", False))
        reward += float(result.reward)
        if hitWall:
            self._rememberWall(attempted)
            wasRevisit = False
            wasReversal = False
            oscillationDetected = False
        else:
            reward += self.shapingPenalty(wasRevisit, wasReversal)
            oscillationDetected = self._recordPositionAndDetectOscillation(cast(tuple[int, int], self.position))
            if oscillationDetected:
                reward += float(getattr(self.config, "oscillationPenalty", 0.0))
        self.state = self.calculateState()
        self._appendCurrentEncodedFrame()
        return StepResult(
            reward=reward,
            done=bool(result.done),
            info={
                "hit_wall": hitWall,
                "was_revisit": wasRevisit,
                "was_reversal": wasReversal,
                "oscillation_detected": oscillationDetected,
            },
            duration=int(result.duration),
        )

    def _applyFlatOption(self, semanticActionId: int, *, training: bool) -> StepResult:
        option = self._actionRegistry.getOption(semanticActionId)
        if option is None:
            raise KeyError(f"Unknown option action_id={semanticActionId}")
        optionName = str(option.name)
        self._optionSelectionCounts[optionName] = self._optionSelectionCounts.get(optionName, 0) + 1
        cumulativeReward = 0.0
        discountedReward = 0.0
        duration = 0
        revisitCount = 0
        reversalCount = 0
        hitWallCount = 0
        done = False
        while not option.shouldTerminate(self._context, duration) and not done:
            primitiveId = option.choosePrimitiveAction(self._context)
            primitiveBefore = self.encodeState(actionSpace=self._primitiveActionSpace()) if self.primitiveBootstrapAgent is not None else None
            primitiveResult = self._applyPrimitiveStep(primitiveId)
            stepReward = float(primitiveResult.reward) + float(option.intrinsicReward(self._context, primitiveResult))
            cumulativeReward += stepReward
            discountedReward += (float(self.config.discountFactor) ** duration) * stepReward
            duration += int(primitiveResult.duration)
            revisitCount += 1 if bool(primitiveResult.info.get("was_revisit", False)) else 0
            reversalCount += 1 if bool(primitiveResult.info.get("was_reversal", False)) else 0
            hitWallCount += 1 if bool(primitiveResult.info.get("hit_wall", False)) else 0
            done = bool(primitiveResult.done)
            if training and self.primitiveBootstrapAgent is not None and primitiveBefore is not None:
                primitiveAfter = self.encodeState(actionSpace=self._primitiveActionSpace())
                self._recordPrimitiveBootstrapTransition(
                    primitiveBefore,
                    primitiveId,
                    primitiveResult,
                    primitiveAfter,
                    done=bool(primitiveResult.done or self.position == self.maze.end),
                )
            if done:
                break
        self._optionDurationTotals[optionName] = self._optionDurationTotals.get(optionName, 0) + max(1, duration)
        if self.position == self.maze.end:
            self._optionSuccessCounts[optionName] = self._optionSuccessCounts.get(optionName, 0) + 1
        return StepResult(
            reward=cumulativeReward,
            done=done,
            info={
                "hit_wall": hitWallCount > 0,
                "was_revisit": revisitCount > 0,
                "was_reversal": reversalCount > 0,
                "option": option.name,
                "action_key": option.name,
            },
            duration=max(1, duration),
            trainingReward=discountedReward,
        )

    def _syncEntitiesFromMaze(self) -> None:
        self._entityRegistry = build_entity_registry(self.maze)
        self._context = EnvironmentContext(
            maze=self.maze,
            bot=self,
            entity_registry=self._entityRegistry,
            action_registry=self._actionRegistry,
            traversal_policy=self._traversal,
        )
        self._context.on(POSITION_CHANGED, self._onPositionChanged)

    def _calculateLocalObservation(self, position: tuple[int, int]) -> tuple[tuple[float, float, float], ...]:
        visited = self.statistics.getVisitedPositions()
        maxDistance = self._observationScale()
        rays: list[tuple[float, float, float]] = []
        for direction in range(4):
            ray = self._sensing.ray_features(
                position,
                direction,
                visited=visited,
                max_distance=maxDistance,
            )
            rays.append(cast(tuple[float, float, float], ray))
        return tuple(rays)

    def _primitiveActionSpace(self) -> LocalActionSpace:
        return self._actionRegistry.buildLocalActionSpace(
            self._context,
            includePrimitives=True,
            includeOptions=False,
        )

    def _optionActionSpace(self) -> LocalActionSpace:
        return self._actionRegistry.buildLocalActionSpace(
            self._context,
            includePrimitives=False,
            includeOptions=True,
        )

    def _flatActionSpace(self) -> LocalActionSpace:
        return self._actionRegistry.buildLocalActionSpace(
            self._context,
            includePrimitives=not self.usesMacroOnlyPolicy,
            includeOptions=getattr(self.config, "useMacroActions", False),
        )

    def _defaultPolicyActionSpace(self) -> LocalActionSpace:
        if self.usesHierarchicalPolicy:
            return self._optionActionSpace()
        return self._flatActionSpace()

    def _resolvePolicyMode(self) -> str:
        if self.usesHierarchicalPolicy:
            return "hierarchical"
        if self.usesMacroOnlyPolicy:
            return "macro_only_dual_record" if self.dualRecordsPrimitivePolicy else "macro_only"
        return "flat"

    def _policyActionCount(self) -> int:
        return self._defaultPolicyActionSpace().numActions

    def _recordPrimitiveBootstrapTransition(
        self,
        before: EncodedState,
        primitiveSemanticId: int,
        primitiveResult: StepResult,
        after: EncodedState,
        *,
        done: bool,
    ) -> None:
        if self.primitiveBootstrapAgent is None:
            return
        primitiveLocalId = self._primitiveActionSpace().localId(primitiveSemanticId)
        if primitiveLocalId is None:
            return
        transition = Transition(
            state=before.values,
            action=primitiveLocalId,
            reward=float(primitiveResult.reward),
            nextState=after.values,
            done=bool(done),
            validActionMask=before.validActionMask,
            nextValidActionMask=after.validActionMask,
            bootstrapDiscount=float(self.config.discountFactor) ** max(1, int(primitiveResult.duration)),
        )
        self.primitiveBootstrapAgent.storeTransition(transition)
        self.primitiveBootstrapAgent.trainStep()
        self.primitiveBootstrapAgent.onEnvironmentStep()
