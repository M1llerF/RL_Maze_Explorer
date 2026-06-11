from __future__ import annotations

import numpy as np
from typing import Any, cast

from botStatistics import BotStatistics
from baseBot import BaseBot
from botTools import BotTools
from bots.common.actions import ActionSpec, DEFAULT_PRIMITIVE_ACTIONS, DEFAULT_ATTACK_ACTIONS
from bots.common.actionRegistry import ActionRegistry
from bots.common.decision import ActionChoice, DecisionInput, LocalActionSpace
from bots.common.optionsLibrary import defaultOptions
from bots.common.stepResult import StepResult
from environment.context import EnvironmentContext
from environment.entityFactory import build_entity_registry
from environment.entityRegistry import EntityRegistry
from environment.movement import GridMovement4Way, buildDefaultMovementExecutors, buildAttackExecutors
from environment.observationEncoder import ObservationEncoder
from environment.sensing import build_sensing_service
from environment.traversal import build_traversal_policy
from .config import QLearningConfig
from bots.botStatus import BotStatus
from bots.dqnlearning.planner import WarmupPlanner
from bots.dqnlearning.warmupCoordinator import WarmupCoordinator
from botFactory import BotCreateContext
from services.episodeRecorder import PostEpisodeRecorder
from services.episodeResult import EpisodePersistencePolicy, EvaluationEpisodeDefinition
from services.repository import ArtifactsRepository
from services.runners import QLearningEpisodeRunner


class QLearning:
    """
    Q-learning algorithm.  Supports both:
      - primitive actions:  standard 1-step Bellman update
      - macro-actions/options: SMDP update (discounted over duration steps)

    Q-table is keyed by (state_key, local_action_id). Local IDs are dense over
    the bot's active policy action space.
    """

    def __init__(
        self,
        qLearningConfig: QLearningConfig,
        num_actions: int = 4,
        repo: ArtifactsRepository | None = None,
        profileName: str | None = None,
    ):
        self.lr = qLearningConfig.learningRate
        self.gamma = qLearningConfig.discountFactor
        self.numActions = num_actions
        self.qTable: dict[tuple[Any, ...], np.ndarray] = {}
        self.initialExplorationRate = float(getattr(qLearningConfig, "epsilonStart", 1.0))
        self.minExplorationRate = float(getattr(qLearningConfig, "epsilonEnd", 0.05))
        self.explorationDecaySteps = max(1, int(getattr(qLearningConfig, "epsilonDecaySteps", 100000)))
        self.totalSteps = 0
        self.usePositionInState = getattr(qLearningConfig, "usePositionInState", True)
        self._repo = repo
        self._profile = profileName

    # ── Action selection ──────────────────────────────────────────────────────

    def chooseAction(self, decision: DecisionInput) -> ActionChoice:
        """Epsilon-greedy action selection, respecting valid action mask."""
        stateKey = self.stateToKey(decision.state)
        if stateKey not in self.qTable:
            self.qTable[stateKey] = np.zeros(self.numActions)

        explorationRate = self.explorationRate()

        valid_ids = [i for i, v in enumerate(decision.action_mask) if v and i < self.numActions]

        if not valid_ids:
            valid_ids = list(range(self.numActions))

        if np.random.rand() < explorationRate:
            return ActionChoice(local_id=int(np.random.choice(valid_ids)))

        q = self.qTable[stateKey]
        # Mask out invalid actions with -inf
        masked_q = np.full(self.numActions, -np.inf)
        for i in valid_ids:
            masked_q[i] = q[i]
        return ActionChoice(local_id=int(np.argmax(masked_q)))

    def explorationRate(self) -> float:
        progress = min(1.0, float(self.totalSteps) / float(self.explorationDecaySteps))
        return max(
            self.minExplorationRate,
            self.initialExplorationRate
            + (self.minExplorationRate - self.initialExplorationRate) * progress,
        )

    # ── Learning updates ──────────────────────────────────────────────────────

    def updateQValue(
        self,
        state: Any,
        action: int,
        reward: float,
        newState: Any,
        next_mask: list[bool] | None = None,
    ) -> None:
        """Standard 1-step Q-learning update (primitive actions)."""
        stateKey = self.stateToKey(state)
        newStateKey = self.stateToKey(newState)
        self._ensureRow(stateKey)
        self._ensureRow(newStateKey)

        future_q = self._maxValidQ(newStateKey, next_mask)
        old_value = self.qTable[stateKey][action]
        target = reward + self.gamma * future_q
        self.qTable[stateKey][action] = old_value + self.lr * (target - old_value)

    def updateSMDP(
        self,
        state: Any,
        action: int,
        cumulative_reward: float,
        duration: int,
        newState: Any,
        next_mask: list[bool] | None = None,
    ) -> None:
        """
        Semi-Markov Decision Process (SMDP) update for macro-actions / options.

        target = cumulative_reward + (gamma^duration) * max Q(s')
        """
        stateKey = self.stateToKey(state)
        newStateKey = self.stateToKey(newState)
        self._ensureRow(stateKey)
        self._ensureRow(newStateKey)

        future_q = self._maxValidQ(newStateKey, next_mask)
        discount = self.gamma ** max(1, duration)
        old_value = self.qTable[stateKey][action]
        target = cumulative_reward + discount * future_q
        self.qTable[stateKey][action] = old_value + self.lr * (target - old_value)

    # ── Persistence ───────────────────────────────────────────────────────────

    def saveQTable(self) -> None:
        if self._repo and self._profile:
            self._repo.saveQTable(self._profile, self.qTable)

    def loadQTable(self) -> None:
        if self._repo and self._profile:
            self.qTable = self._repo.loadQTable(self._profile) or {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def stateToKey(self, state: Any) -> tuple[Any, ...]:
        if not isinstance(state, (tuple, list)):
            return (state,)
        if len(state) < 4:
            return tuple(state)
        positionIndex, wallDistances, goalDirection, *rest = state
        entityFeatures = tuple(rest[0]) if rest else ()
        if self.usePositionInState:
            return positionIndex, wallDistances, goalDirection, entityFeatures
        return wallDistances, goalDirection, entityFeatures

    def _ensureRow(self, key: tuple[Any, ...]) -> None:
        if key not in self.qTable:
            self.qTable[key] = np.zeros(self.numActions)

    def _maxValidQ(self, key: tuple[Any, ...], mask: list[bool] | None) -> float:
        self._ensureRow(key)
        q = self.qTable[key]
        if mask is not None:
            valid_ids = [i for i, v in enumerate(mask) if v and i < len(q)]
            if valid_ids:
                return float(np.max(q[valid_ids]))
        return float(np.max(q))


class QLearningBot(BaseBot):
    def __init__(self, ctx: BotCreateContext) -> None:
        maze = ctx.maze
        config = cast(QLearningConfig, ctx.config)
        rewardSystem = ctx.rewardSystem
        statistics = cast(BotStatistics, ctx.statistics)
        profileName = ctx.profileName
        repository = ctx.repository

        cast(Any, super()).__init__(maze, statistics, config)
        self.maze: Any = maze
        self.statistics: Any = statistics
        self.repo = repository
        self.rewardSystem = rewardSystem
        self.profileName = profileName
        self.totalReward = 0.0
        self.episodeCounter = 0
        self.currentEpisodeSteps = 0
        self.position = maze.getStart()
        self.previousPosition: tuple[int, int] | None = None
        self.lastAction: int | None = None
        self.tools = BotTools(maze, profileName)
        self._traversal = build_traversal_policy(maze, profileName)
        self._sensing = build_sensing_service(maze, profileName)

        # ── Action system setup ───────────────────────────────────────────────
        self._entityRegistry = EntityRegistry()
        self._movementModel = GridMovement4Way()
        self._context = EnvironmentContext(
            maze=maze,
            bot=self,
            entity_registry=self._entityRegistry,
        )
        self._actionRegistry = ActionRegistry()
        executors = buildDefaultMovementExecutors(self._movementModel)
        for spec in DEFAULT_PRIMITIVE_ACTIONS:
            self._actionRegistry.registerPrimitive(spec, executors[spec.id])
        if getattr(config, "useAttackActions", False):
            attack_executors = buildAttackExecutors()
            for spec in DEFAULT_ATTACK_ACTIONS:
                self._actionRegistry.registerPrimitive(spec, attack_executors[spec.id])
        if getattr(config, "useMacroActions", False):
            for option in defaultOptions():
                self._actionRegistry.registerOption(option)
        # Wire context → registry (circular reference resolved after registry is built)
        self._context.setActionRegistry(self._actionRegistry)
        self._observationEncoder = ObservationEncoder(scale=float(max(maze.width, maze.height, 1)))
        self._syncEntitiesFromMaze()

        self._knownOpen: set[tuple[int, int]] = set()
        self._knownWalls: set[tuple[int, int]] = set()
        self._seenGoals: set[tuple[int, int]] = set()
        self._collectingWarmup: bool = False

        self.qLearning = QLearning(
            config,
            num_actions=self._policyActionCount(),
            repo=self.repo,
            profileName=profileName,
        )
        self.state = self.calculateState()
        self.qLearning.loadQTable()

        _planner = WarmupPlanner(
            maze=self.maze,
            profileName=self.profileName,
            tools=self.tools,
            knownOpen=self._knownOpen,
            knownWalls=self._knownWalls,
            seenGoals=self._seenGoals,
        )
        self.warmupCoordinator = WarmupCoordinator(_planner)
        self._observePosition(self.position)

        self.repo.ensureMazeFile(profileName)
        mazeData = self.repo.loadMazeData(profileName)
        self.highestReward = float(mazeData.get("highest", {}).get("reward", float("-inf")))
        self.lowestReward = float(mazeData.get("lowest", {}).get("reward", float("inf")))

        # Visualization state
        self._visActive = False
        self._visInitialized = False
        self._visOptimalPath: list[Any] | None = None
        self._visOptimalLength = 0
        self._visStepLimit = 0
        self._visBestDistance = 0
        self._visNoProgressSteps = 0
        self._visProgressPatience = 0
        self._visSteps = 0
        self._visTimesHitWall = 0

        self.runner = QLearningEpisodeRunner(self)
        self.recorder = PostEpisodeRecorder(self.repo)

    # ── BaseBot contract ──────────────────────────────────────────────────────

    def getActionSpace(self) -> list[ActionSpec]:
        return self._actionRegistry.getActionSpace()

    def getPolicyActionSpace(self) -> list[ActionSpec]:
        return self._buildPolicyActionSpace().actionList()

    def getValidActionMask(self) -> list[bool]:
        return self._buildPolicyActionSpace().maskList()

    def buildDecisionInput(self) -> DecisionInput:
        return DecisionInput(
            state=self.state,
            actionSpace=self._buildPolicyActionSpace(),
            level="flat",
        )

    def applyStep(self, action_id: int) -> StepResult:
        """Execute action through ActionRegistry and return StepResult."""
        actionSpace = self._buildPolicyActionSpace()
        semanticActionId = self._actionRegistry.semanticIdForLocal(actionSpace, action_id)
        self.lastAction = int(semanticActionId)
        if self._actionRegistry.isOption(semanticActionId):
            option = self._actionRegistry.getOption(semanticActionId)
            if option is None:
                raise KeyError(f"Unknown option action_id={semanticActionId}")
            cumulativeReward = 0.0
            discountedReward = 0.0
            duration = 0
            revisitCount = 0
            reversalCount = 0
            hitWallCount = 0
            done = False
            while not option.shouldTerminate(self._context, duration) and not done:
                primitiveId = option.choosePrimitiveAction(self._context)
                startPos = cast(tuple[int, int], self.position)
                attempted = self._movementModel.candidatePosition(startPos, primitiveId)
                wasReversal = self.previousPosition is not None and attempted == self.previousPosition
                wasRevisit = attempted in self.statistics.getVisitedPositions()
                rewardContext = self.rewardSystem.buildStepContext(
                    startPos,
                    attempted,
                    cast(list[tuple[int, int]], getattr(self, "_currentOptimalPath", [])),
                    int(getattr(self, "_currentOptimalLength", 0)),
                    self.statistics.getVisitedPositions(),
                )
                envReward = float(
                    self.rewardSystem.evaluateStep(rewardContext)
                    if hasattr(self.rewardSystem, "evaluateStep")
                    else self.rewardSystem.getReward(
                        startPos,
                        attempted,
                        cast(list[tuple[int, int]], getattr(self, "_currentOptimalPath", [])),
                        int(getattr(self, "_currentOptimalLength", 0)),
                        self.statistics.getVisitedPositions(),
                    )
                )
                primitiveResult = self._actionRegistry.executePrimitive(primitiveId, self._context)
                hitWall = bool(primitiveResult.info.get("hit_wall", False))
                self._observePosition(self.position)
                if hitWall:
                    self.warmupCoordinator.onWallDiscovered()
                    wasReversal = False
                    wasRevisit = False
                stepReward = (
                    envReward
                    + float(primitiveResult.reward)
                    + float(option.intrinsicReward(self._context, primitiveResult))
                )
                if not hitWall:
                    stepReward += self.shapingPenalty(wasRevisit, wasReversal)
                cumulativeReward += stepReward
                discountedReward += (self.qLearning.gamma ** duration) * stepReward
                duration += 1
                revisitCount += 1 if wasRevisit else 0
                reversalCount += 1 if wasReversal else 0
                hitWallCount += 1 if hitWall else 0
                done = bool(primitiveResult.done or self.position == self.maze.end)
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
                duration=duration,
                trainingReward=discountedReward,
            )
        result = self._actionRegistry.executePrimitive(semanticActionId, self._context)
        self._observePosition(self.position)
        if bool(result.info.get("hit_wall", False)):
            self.warmupCoordinator.onWallDiscovered()
        self.state = self.calculateState()
        return result

    def addReward(self, amount: float, reason: str = "") -> None:
        self.totalReward += amount

    def shapingPenalty(self, wasRevisit: bool = False, wasReversal: bool = False) -> float:
        penalty = 0.0
        if wasRevisit:
            penalty += float(getattr(self.config, "repeatVisitPenaltyScale", 0.0))
        if wasReversal:
            penalty += float(getattr(self.config, "immediateReversalPenalty", 0.0))
        return penalty

    @property
    def isWarmingUp(self) -> bool:
        return bool(self._collectingWarmup)

    @property
    def usesMacroOnlyPolicy(self) -> bool:
        return getattr(self.config, "useMacroOnlyPolicy", False)

    # ── State / observation ───────────────────────────────────────────────────

    def calculateState(self, position: tuple[int, int] | None = None) -> tuple[Any, ...]:
        return self._observationEncoder.encodeCompactState(
            self._context,
            self._sensing,
            position=position,
            include_entity_features=getattr(self.config, "useEntityObservation", False),
            include_enemy_features=getattr(self.config, "useEnemyObservation", False),
        )

    def encodeState(self) -> tuple[Any, ...]:
        return self.calculateState()

    def _observePosition(self, position: tuple[int, int]) -> None:
        goal = self.maze.end
        for row, col in self._sensing.visible_open_cells(position):
            self._knownOpen.add((row, col))
            self._knownWalls.discard((row, col))
            if (row, col) == goal:
                self._seenGoals.add((row, col))
        for row, col in self._sensing.first_blocked_cells(position):
            if (row, col) not in self._knownOpen:
                self._knownWalls.add((row, col))

    def selectWarmupAction(self, decision: Any, plannerLocal: int | None) -> ActionChoice:
        mask = list(decision.action_mask)
        if plannerLocal is not None and 0 <= plannerLocal < len(mask) and mask[plannerLocal]:
            return ActionChoice(local_id=int(plannerLocal))
        valid_ids = [i for i, v in enumerate(mask) if v]
        if not valid_ids:
            valid_ids = list(range(self._policyActionCount()))
        return ActionChoice(local_id=int(np.random.choice(valid_ids)))

    def getBotSpecificData(self) -> dict[str, Any]:
        return {"q_table": self.qLearning.qTable}

    def initializeSpecificData(self, data: dict[str, Any]) -> None:
        self.qLearning.qTable = data.get("q_table", {})
        self.qLearning.loadQTable()

    # ── Episode management ────────────────────────────────────────────────────

    def runEpisode(self) -> None:
        result = self.runner.runEpisode()
        self.recorder.record(self, result)

    def applyLearningUpdate(
        self,
        startState: Any,
        localId: int,
        reward: float,
        duration: int = 1,
    ) -> None:
        """Apply Q-learning update after a step. Called by QLearningEpisodeRunner."""
        new_state = self.calculateState()
        next_mask = self.getValidActionMask()
        if duration > 1:
            self.qLearning.updateSMDP(startState, localId, reward, duration, new_state, next_mask)
        else:
            self.qLearning.updateQValue(startState, localId, reward, new_state, next_mask)
        self.qLearning.totalSteps += duration
        self.state = new_state

    def persistTrainingArtifacts(self) -> None:
        self.qLearning.saveQTable()

    def getEvaluationEpisodeDefinition(self) -> EvaluationEpisodeDefinition:
        return EvaluationEpisodeDefinition(
            description=(
                "Greedy tabular evaluation with learning disabled and epsilon forced to 0.0. "
                "This bot does not schedule dedicated evaluation episodes during training."
            ),
            frequency=0,
            is_dedicated_episode=False,
            eval_epsilon=0.0,
            persistence=EpisodePersistencePolicy(
                save_maze_episode=False,
                save_heatmap_stats=False,
                append_reward=False,
            ),
        )

    # ── BaseBot public contract implementations ───────────────────────────────

    def isOptionAction(self, semanticId: int) -> bool:
        return self._actionRegistry.isOption(semanticId)

    def executePrimitiveAction(self, semanticId: int) -> Any:
        return self._actionRegistry.executePrimitive(semanticId, self._context)

    def selectAction(self, decision: Any) -> Any:
        warmupChoice = self.warmupCoordinator.selectAction(self, decision, training=True)
        if warmupChoice is not None:
            return warmupChoice
        autoAttackLocalId = self.autoAttackAdjacentEnemyLocalId(decision.actionSpace)
        if autoAttackLocalId is not None:
            return ActionChoice(local_id=autoAttackLocalId)
        return self.qLearning.chooseAction(decision)

    def getStatus(self) -> BotStatus:
        explorationRate = float(self.qLearning.explorationRate())
        return BotStatus(
            episode_count=int(self.episodeCounter),
            is_warming_up=bool(self._collectingWarmup),
            replay_size=0,
            warmup_required=0,
            current_exploration_rate=explorationRate,
            epsilon_is_manual=False,
            checkpoint_frequency=1,
            current_episode_steps=int(self.currentEpisodeSteps),
            last_episode_success=getattr(self, "lastEpisodeSuccess", False),
            artifact_save_label="Q-table",
        )

    def reset(self) -> None:
        self.position = self.maze.start
        self.previousPosition = None
        self.lastAction = None
        self.statistics.reset()
        self.totalReward = 0.0
        self.currentEpisodeSteps = 0
        self._pushCooldownRemaining = 0
        self._knownOpen.clear()
        self._knownWalls.clear()
        self._seenGoals.clear()
        self.warmupCoordinator.reset()
        self._syncEntitiesFromMaze()
        self.state = self.calculateState()
        self._observePosition(self.position)

    def onEpisodeStart(self, mode: str) -> None:
        if mode not in {"training", "visualization", "evaluation"}:
            raise ValueError(f"Unsupported episode mode for QLearningBot: {mode}")

    def onEpisodeStep(self, mode: str, stepIndex: int) -> None:
        if mode not in {"training", "visualization", "evaluation"}:
            raise ValueError(f"Unsupported episode mode for QLearningBot: {mode}")
        if stepIndex < 0:
            raise ValueError(f"step_index must be non-negative, got {stepIndex}")

    def onEpisodeEnd(self, mode: str, outcome: str) -> None:
        if mode not in {"training", "visualization", "evaluation"}:
            raise ValueError(f"Unsupported episode mode for QLearningBot: {mode}")
        if not outcome:
            raise ValueError("outcome must be a non-empty string")

    # ── Visualization ─────────────────────────────────────────────────────────

    def beginVisualizationEpisode(self) -> None:
        self.runner.beginVisualizationEpisode()

    def stepVisualization(self, maxSteps: int = 1) -> bool:
        return bool(self.runner.stepVisualization(maxSteps=maxSteps))

    def _finalizeVisualizationEpisode(self) -> bool:
        self._visActive = False
        self._visInitialized = False
        return True

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _syncEntitiesFromMaze(self) -> None:
        self._entityRegistry = build_entity_registry(self.maze)
        self._context = EnvironmentContext(
            maze=self.maze,
            bot=self,
            entity_registry=self._entityRegistry,
            traversal_policy=self._traversal,
        )
        self._context.setActionRegistry(self._actionRegistry)

    def _buildPolicyActionSpace(self) -> LocalActionSpace:
        return self._actionRegistry.buildLocalActionSpace(
            self._context,
            includePrimitives=not getattr(self.config, "useMacroOnlyPolicy", False),
            includeOptions=getattr(self.config, "useMacroActions", False),
        )

    def _policyActionCount(self) -> int:
        return self._buildPolicyActionSpace().numActions
