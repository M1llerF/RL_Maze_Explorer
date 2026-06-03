from __future__ import annotations

from typing import Any, cast

import numpy as np

from baseBot import BaseBot
from botTools import BotTools
from services.repository import ArtifactsRepository
from services.runners import DQNEpisodeRunner

from .agent import DqnAgent
from .checkpoint import CheckpointIO, CheckpointMeta
from .config import DQNConfig
from .encoder import StateEncoder
from .planner import WarmupPlanner
from .replay import UniformReplayStore
from .types import EncodedState, Transition


class DQNBot(BaseBot):
    config: DQNConfig  # narrows BaseBot's Any for type checkers

    def __init__(
        self,
        maze: Any,
        config: DQNConfig,
        rewardSystem: Any,
        statistics: Any,
        profileName: str,
        repository: ArtifactsRepository | None = None,
        loadCheckpoint: bool = True,
    ) -> None:
        super().__init__(maze, statistics, config)
        self.rewardSystem = rewardSystem
        self.profileName = profileName
        self.repo = repository or ArtifactsRepository()
        self.tools = BotTools(maze)
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
        self._syncMazeDependentState()
        self._observePosition(cast(tuple[int, int], self.position))
        self.state = self.calculateState()
        schema = self.encoder.inferSchema(self.state)
        self.agent = DqnAgent(
            self.config,
            inputDim=schema.inputDim,
            flatDim=schema.flatDim,
            mapShape=schema.mapShape,
        )
        self.checkpoint = CheckpointIO(self.repo, self.profileName, self.agent.device)
        self.checkpointMeta = CheckpointMeta(
            stateSchemaVersion=schema.stateSchemaVersion,
            encoderConfigFingerprint=schema.encoderConfigFingerprint,
            inputDim=schema.inputDim,
        )
        self._planner = WarmupPlanner(
            maze=self.maze,
            profileName=self.profileName,
            tools=self.tools,
            knownOpen=self._knownOpen,
            knownWalls=self._knownWalls,
            seenGoals=self._seenGoals,
        )
        self.runner = DQNEpisodeRunner(self)
        self._visInitialized = False
        if loadCheckpoint:
            self._loadCheckpoint()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._syncMazeDependentState()
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
        self.state = self.calculateState()
        self._planner.reset()
        self._observePosition(cast(tuple[int, int], self.position))

    def _syncMazeDependentState(self) -> None:
        mazeH = max(1, int(self.maze.height))
        mazeW = max(1, int(self.maze.width))
        if self._knownMap.shape != (mazeH, mazeW, 3):
            self._knownMap = np.zeros((mazeH, mazeW, 3), dtype=np.float32)
        if self._visitedMap.shape != (mazeH, mazeW):
            self._visitedMap = np.zeros((mazeH, mazeW), dtype=np.float32)
        self.encoder = StateEncoder(self.config, mazeHeight=mazeH, mazeWidth=mazeW)

    # ------------------------------------------------------------------
    # Episode hooks (BaseBot interface)
    # ------------------------------------------------------------------

    def runEpisode(self) -> None:
        self.runner.runEpisode()

    def onEpisodeStart(self, mode: str) -> None:
        if mode != "training":
            raise ValueError(f"Unsupported episode mode: {mode}")

    def onEpisodeStep(self, mode: str, stepIndex: int) -> None:
        if mode != "training":
            raise ValueError(f"Unsupported episode mode: {mode}")
        if stepIndex < 0:
            raise ValueError("stepIndex must be non-negative")

    def onEpisodeEnd(self, mode: str, outcome: str) -> None:
        if mode != "training":
            raise ValueError(f"Unsupported episode mode: {mode}")
        if not outcome:
            raise ValueError("outcome must be non-empty")

    # ------------------------------------------------------------------
    # Public step interface (used by DQNEpisodeRunner)
    # ------------------------------------------------------------------

    @property
    def isWarmingUp(self) -> bool:
        return self._collectingWarmup

    def applyStep(self, action: int) -> tuple[bool, bool, bool, tuple[Any, ...]]:
        """
        Apply action and update all internal state.

        Returns (hitWall, wasRevisit, wasReversal, nextState).
        Callers must compute reward BEFORE calling this when they need the
        pre-step visited set (getReward uses visited positions before this step).
        """
        currentPos = cast(tuple[int, int], self.position)
        attempted = self.tools.calculateNextPosition(currentPos, action)
        if not self.maze.isValidPosition(self.profileName, attempted[0], attempted[1]):
            self._rememberWall(attempted)
            return True, False, False, self.calculateState()

        wasRevisit = attempted in self.statistics.getVisitedPositions()
        wasReversal = self.previousPosition is not None and attempted == self.previousPosition
        if wasRevisit:
            self.statistics.timesRevisitedSquares += 1
        else:
            self.statistics.nonRepeatingStepsTaken += 1
        self.statistics.updateLastVisited(currentPos)
        self.previousPosition = currentPos
        self.position = attempted
        self.statistics.updateVisitedPositions(self.position)
        self._observePosition(self.position)
        self.state = self.calculateState()
        return False, wasRevisit, wasReversal, self.state

    def addReward(self, reward: float) -> None:
        self.totalReward += reward

    def shapingPenalty(self, wasRevisit: bool, wasReversal: bool) -> float:
        """Config-driven per-step shaping penalties, consolidated in one place."""
        penalty = 0.0
        if wasRevisit:
            penalty -= float(self.config.repeatVisitPenaltyScale)
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

    def calculateState(self) -> tuple[Any, ...]:
        position = cast(tuple[int, int], self.position)
        wallDistances, _ = self.tools.detectWalls(position)
        previousDelta = (
            (0, 0)
            if self.previousPosition is None
            else (self.previousPosition[0] - position[0], self.previousPosition[1] - position[1])
        )
        validActions = tuple(
            1 if self.maze.isValidPosition(self.profileName, *self.tools.calculateNextPosition(position, a)) else 0
            for a in range(4)
        )
        # Local observation: first-person line-of-sight rays per direction.
        localObservation = self._calculateLocalObservation(position)
        neuralMap: tuple[float, ...] | np.ndarray = ()
        if self.config.useRichEncoding:
            neuralMap = self.encoder.encodeNeuralMap(position, self._knownMap, self._visitedMap)
        # Goal direction: relative position when seen, zero-vector while unknown.
        goal = cast(tuple[int, int], self.maze.end)
        goalSeen = 1.0 if goal in self._seenGoals else 0.0
        observationScale = self._observationScale()
        goalDr = float(np.clip(float(goal[0] - position[0]) / observationScale, -1.0, 1.0)) if goalSeen else 0.0
        goalDc = float(np.clip(float(goal[1] - position[1]) / observationScale, -1.0, 1.0)) if goalSeen else 0.0
        goalInfo: tuple[float, float, float] = (goalDr, goalDc, goalSeen)
        return (
            position,
            wallDistances,
            previousDelta,
            -1 if self.lastAction is None else self.lastAction,
            localObservation,
            neuralMap,
            validActions,
            goalInfo,
        )

    def encodeState(self, state: tuple[Any, ...] | None = None) -> EncodedState:
        return self.encoder.encode(self.state if state is None else state)

    def chooseAction(self, encodedState: EncodedState, training: bool) -> int:
        if training and self._collectingWarmup:
            plannerAction = self._planner.nextAction(cast(tuple[int, int], self.position))
            return self.agent.selectWarmupAction(encodedState, plannerAction)
        return self.agent.chooseAction(encodedState, training=training)

    def makeTransition(
        self,
        encodedState: EncodedState,
        action: int,
        reward: float,
        nextEncodedState: EncodedState,
        done: bool,
    ) -> Transition:
        return Transition(
            state=encodedState.values,
            action=action,
            reward=reward,
            nextState=nextEncodedState.values,
            done=done,
            validActionMask=encodedState.validActionMask,
            nextValidActionMask=nextEncodedState.validActionMask,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def saveCheckpoint(self) -> None:
        freq = int(getattr(self.config, "checkpointFrequency", 10))
        if freq > 0 and self.episodeCounter % freq != 0:
            return
        # The replay buffer is not persisted — serialising 15 k × 1600-float
        # transitions to disk on every save blocks the training thread for several
        # seconds and is the primary cause of the "steps frozen" symptom.
        # Model weights are small (~100 KB) and save in < 1 ms.
        self.checkpoint.save(
            policyStateDict=self.agent.policy.state_dict(),
            targetStateDict=self.agent.target.state_dict(),
            optimizerStateDict=self.agent.optimizer.state_dict(),
            globalStep=self.agent.globalStep,
            episodeCount=self.episodeCounter,
            meta=self.checkpointMeta,
            replayState=None,
        )

    def _loadCheckpoint(self) -> None:
        checkpoint, reason = self.checkpoint.loadWithReason(self.checkpointMeta)
        if checkpoint is None:
            print(f"[DQN] checkpoint not loaded for profile '{self.profileName}': {reason}")
            return
        self.agent.policy.load_state_dict(checkpoint["policy"])
        self.agent.target.load_state_dict(checkpoint["target"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])
        self.agent.globalStep = int(checkpoint.get("global_step", 0))
        self.episodeCounter = int(checkpoint.get("episode_count", 0))
        replayState = checkpoint.get("replay")
        if replayState is not None and isinstance(self.agent.replay, UniformReplayStore):
            self.agent.replay.load_serializable(cast(dict[str, Any], replayState))
        compatibilityNote = " [compat]" if reason == "ok_compat_fingerprint" else ""
        print(
            f"[DQN] checkpoint loaded{compatibilityNote} for profile '{self.profileName}' "
            f"(globalStep={self.agent.globalStep}, replaySize={len(self.agent.replay)})"
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def beginVisualizationEpisode(self) -> None:
        self.reset()
        self._visInitialized = True

    def stepVisualization(self, maxSteps: int = 1) -> bool:
        if not self._visInitialized:
            self.beginVisualizationEpisode()
        for _ in range(maxSteps):
            if self.position == self.maze.end:
                self._visInitialized = False
                return True
            encoded = self.encodeState()
            action = self.agent.chooseAction(encoded, training=False)
            newPosition = self.tools.calculateNextPosition(cast(tuple[int, int], self.position), action)
            if self.maze.isValidPosition(self.profileName, newPosition[0], newPosition[1]):
                self.previousPosition = cast(tuple[int, int], self.position)
                self.position = newPosition
                self.statistics.updateVisitedPositions(self.position)
                self._observePosition(self.position)
            else:
                self._rememberWall(newPosition)
            self.state = self.calculateState()
        return False

    # ------------------------------------------------------------------
    # Observation helpers (private — called only from within this class)
    # ------------------------------------------------------------------

    def _observePosition(self, position: tuple[int, int]) -> None:
        self._knownOpen.add(position)
        self._knownWalls.discard(position)
        self._knownMap[position[0], position[1], 0] = 1.0
        self._knownMap[position[0], position[1], 1] = 0.0
        self._visitedMap[position[0], position[1]] = 1.0
        goal = cast(tuple[int, int], self.maze.end)
        if position == goal:
            self._seenGoals.add(position)
            self._knownMap[position[0], position[1], 2] = 1.0
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row, col = position
            while True:
                row += drow
                col += dcol
                candidate = (row, col)
                if not (0 <= row < int(self.maze.height) and 0 <= col < int(self.maze.width)):
                    break
                if self.maze.isValidPosition(self.profileName, row, col):
                    self._knownOpen.add(candidate)
                    self._knownWalls.discard(candidate)
                    self._knownMap[row, col, 0] = 1.0
                    self._knownMap[row, col, 1] = 0.0
                    if candidate == goal:
                        self._seenGoals.add(candidate)
                        self._knownMap[row, col, 2] = 1.0
                else:
                    if candidate not in self._knownOpen:
                        self._knownWalls.add(candidate)
                        self._knownMap[row, col, 1] = 1.0
                    break

    def _rememberWall(self, position: tuple[int, int]) -> None:
        if position not in self._knownOpen:
            self._knownWalls.add(position)
            if 0 <= position[0] < int(self.maze.height) and 0 <= position[1] < int(self.maze.width):
                self._knownMap[position[0], position[1], 1] = 1.0
        self._planner.onWallDiscovered()

    def _calculateLocalObservation(self, position: tuple[int, int]) -> tuple[tuple[float, float, float], ...]:
        visited = self.statistics.getVisitedPositions()
        maxDistance = self._observationScale()
        rays: list[tuple[float, float, float]] = []
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row, col = position
            distance = 0
            goalVisible = 0.0
            visitedVisible = 0.0
            while True:
                row += drow
                col += dcol
                if not (0 <= row < int(self.maze.height) and 0 <= col < int(self.maze.width)):
                    break
                if not self.maze.isValidPosition(self.profileName, row, col):
                    break
                distance += 1
                if (row, col) == cast(tuple[int, int], self.maze.end):
                    goalVisible = 1.0
                if visited.get((row, col), 0) > 0:
                    visitedVisible = 1.0
            rays.append((float(np.clip(float(distance) / maxDistance, 0.0, 1.0)), goalVisible, visitedVisible))
        return tuple(rays)
