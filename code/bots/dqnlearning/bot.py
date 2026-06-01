from __future__ import annotations

from typing import Any, cast

from baseBot import BaseBot
from botTools import BotTools
from pathfinding import Pathfinding
from services.repository import ArtifactsRepository
from services.runners import DQNEpisodeRunner

from .agent import DqnAgent
from .checkpoint import CheckpointIO, CheckpointMeta
from .config import DQNConfig
from .encoder import StateEncoder
from .types import EncodedState, Transition
from .warmup import PlannerObservationContext, PlannerWarmupPolicy


class DQNBot(BaseBot):
    def __init__(
        self,
        maze: Any,
        config: DQNConfig,
        rewardSystem: Any,
        statistics: Any,
        profileName: str,
        repository: ArtifactsRepository | None = None,
    ) -> None:
        cast(Any, super()).__init__(maze, statistics, config)
        self.maze = maze
        self.config = config
        self.rewardSystem = rewardSystem
        self.statistics = statistics
        self.profileName = profileName
        self.repo = repository or ArtifactsRepository()
        self.tools = BotTools(maze)
        self.position = self.maze.getStart()
        self.previousPosition: tuple[int, int] | None = None
        self.lastAction: int | None = None
        self.totalReward = 0.0
        self.episodeCounter = 0
        self.state = self.calculateState()
        self.encoder = StateEncoder(self.config, mazeHeight=int(self.maze.height), mazeWidth=int(self.maze.width))
        schema = self.encoder.inferSchema(self.state)
        self.agent = DqnAgent(self.config, inputDim=schema.inputDim)
        self.checkpoint = CheckpointIO(self.repo, self.profileName, self.agent.device)
        self.checkpointMeta = CheckpointMeta(
            stateSchemaVersion=schema.stateSchemaVersion,
            encoderConfigFingerprint=schema.encoderConfigFingerprint,
            inputDim=schema.inputDim,
        )
        self.warmupPolicy = PlannerWarmupPolicy()
        self.runner = DQNEpisodeRunner(self)
        self._visInitialized = False
        self._knownOpen: set[tuple[int, int]] = set()
        self._knownWalls: set[tuple[int, int]] = set()
        self._seenGoals: set[tuple[int, int]] = set()
        self._warmupPlanActions: list[int] = []
        self._warmupPlanIndex = 0
        self._dstarPlanner: Any | None = None
        self._dstarGoal: tuple[int, int] | None = None
        self._dstarKnownBlocked: set[tuple[int, int]] = set()
        self._dstarPlannerDirty = True
        self._frontierRefreshIntervalSteps = 6
        self._stepsSinceFrontierRefresh = self._frontierRefreshIntervalSteps
        self._observePosition(cast(tuple[int, int], self.position))
        self._loadCheckpoint()

    def reset(self) -> None:
        self.position = self.maze.getStart()
        self.previousPosition = None
        self.lastAction = None
        self.statistics.reset()
        self.statistics.updateVisitedPositions(self.position)
        self.totalReward = 0.0
        self.state = self.calculateState()
        self._knownOpen.clear()
        self._knownWalls.clear()
        self._seenGoals.clear()
        self._warmupPlanActions = []
        self._warmupPlanIndex = 0
        self._dstarPlanner = None
        self._dstarGoal = None
        self._dstarKnownBlocked.clear()
        self._dstarPlannerDirty = True
        self._stepsSinceFrontierRefresh = self._frontierRefreshIntervalSteps
        self._observePosition(cast(tuple[int, int], self.position))

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
        localObservation: tuple[tuple[float, float, float], ...] = ()
        neuralMap: tuple[float, ...] = ()
        if self.config.useRichEncoding:
            localObservation = self._calculateLocalObservation(position)
            neuralMap = self._encodeNeuralMap(position)
        return (
            position,
            wallDistances,
            previousDelta,
            -1 if self.lastAction is None else self.lastAction,
            localObservation,
            neuralMap,
            validActions,
        )

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

    def encodeState(self, state: tuple[Any, ...] | None = None) -> EncodedState:
        return self.encoder.encode(self.state if state is None else state)

    def chooseAction(self, encodedState: EncodedState, training: bool) -> int:
        if training and self.config.warmupEnabled and len(self.agent.replay) < self.config.replayWarmupSteps:
            plannerAction = self._plannerAction()
            context = PlannerObservationContext(
                plannerActions=() if plannerAction is None else (plannerAction,),
                validActionMask=tuple(bool(v) for v in encodedState.validActionMask),
            )
            selected = self.warmupPolicy.selectAction(context)
            return self.agent.selectWarmupAction(encodedState, selected)
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

    def saveCheckpoint(self) -> None:
        self.checkpoint.save(
            policyStateDict=self.agent.policy.state_dict(),
            targetStateDict=self.agent.target.state_dict(),
            optimizerStateDict=self.agent.optimizer.state_dict(),
            globalStep=self.agent.globalStep,
            episodeCount=self.episodeCounter,
            meta=self.checkpointMeta,
        )

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
                self.statistics.updateVisitedPositions(cast(tuple[int, int], self.position))
                self._observePosition(cast(tuple[int, int], self.position))
            else:
                self._rememberWall(newPosition)
            self.state = self.calculateState()
        return False

    def _plannerAction(self) -> int | None:
        current = cast(tuple[int, int], self.position)
        goal = cast(tuple[int, int], self.maze.end)
        goalKnown = goal in self._seenGoals
        for _ in range(4):
            if goalKnown:
                action = self._nextActionFromPersistentDstar(current, goal)
                if action is not None:
                    nextPosition = self.tools.calculateNextPosition(current, action)
                    if self.maze.isValidPosition(self.profileName, nextPosition[0], nextPosition[1]):
                        return action
                    self._rememberWall(nextPosition)
                    continue

            shouldRefreshFrontier = (
                self._warmupPlanIndex >= len(self._warmupPlanActions)
                or self._dstarPlannerDirty
                or self._stepsSinceFrontierRefresh >= self._frontierRefreshIntervalSteps
            )
            if shouldRefreshFrontier:
                self._warmupPlanActions = self.tools.getDynamicWarmupActions(
                    current,
                    goal,
                    maxSteps=64,
                    knownOpen=set(self._knownOpen),
                    knownWalls=set(self._knownWalls),
                    seenGoals=set(self._seenGoals),
                )
                self._warmupPlanIndex = 0
                self._stepsSinceFrontierRefresh = 0
            if self._warmupPlanIndex < len(self._warmupPlanActions):
                action = int(self._warmupPlanActions[self._warmupPlanIndex])
                self._warmupPlanIndex += 1
                self._stepsSinceFrontierRefresh += 1
                nextPosition = self.tools.calculateNextPosition(current, action)
                if self.maze.isValidPosition(self.profileName, nextPosition[0], nextPosition[1]):
                    return action
                self._rememberWall(nextPosition)
                continue
            break
        return None

    def _nextActionFromPersistentDstar(self, current: tuple[int, int], goal: tuple[int, int]) -> int | None:
        if self._dstarPlanner is None or self._dstarPlannerDirty or self._dstarGoal != goal:
            self._dstarPlanner = Pathfinding._DStarLitePlanner(
                height=int(self.maze.height),
                width=int(self.maze.width),
                start=current,
                goal=goal,
                blocked=set(self._knownWalls),
            )
            self._dstarGoal = goal
            self._dstarKnownBlocked = set(self._knownWalls)
            self._dstarPlannerDirty = False
        else:
            self._dstarPlanner.moveStart(current)
            newBlocked = set(self._knownWalls) - self._dstarKnownBlocked
            if newBlocked:
                self._dstarPlanner.updateBlockedCells(newBlocked)
                self._dstarKnownBlocked.update(newBlocked)

        nxt = self._dstarPlanner.nextStep()
        if nxt is None:
            return None
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        return {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}.get(delta)

    def _loadCheckpoint(self) -> None:
        checkpoint = self.checkpoint.load(self.checkpointMeta)
        if checkpoint is None:
            return
        self.agent.policy.load_state_dict(checkpoint["policy"])
        self.agent.target.load_state_dict(checkpoint["target"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])
        self.agent.globalStep = int(checkpoint.get("global_step", 0))
        self.episodeCounter = int(checkpoint.get("episode_count", 0))

    def _observePosition(self, position: tuple[int, int]) -> None:
        self._knownOpen.add(position)
        self._knownWalls.discard(position)
        if position == cast(tuple[int, int], self.maze.end):
            self._seenGoals.add(position)
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
                    if candidate == cast(tuple[int, int], self.maze.end):
                        self._seenGoals.add(candidate)
                else:
                    if candidate not in self._knownOpen:
                        self._knownWalls.add(candidate)
                    break

    def _rememberWall(self, position: tuple[int, int]) -> None:
        if position not in self._knownOpen:
            self._knownWalls.add(position)
        # Cached warmup plans become stale whenever a new wall is discovered.
        self._warmupPlanActions = []
        self._warmupPlanIndex = 0
        # Keep planner state incremental; blocked cells are injected on next step.
        self._stepsSinceFrontierRefresh = self._frontierRefreshIntervalSteps

    def _calculateLocalObservation(self, position: tuple[int, int]) -> tuple[tuple[float, float, float], ...]:
        visited = self.statistics.getVisitedPositions()
        maxDistance = float(max(1, int(self.maze.width), int(self.maze.height)))
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
            rays.append((float(distance) / maxDistance, goalVisible, visitedVisible))
        return tuple(rays)

    def _encodeNeuralMap(self, position: tuple[int, int]) -> tuple[float, ...]:
        mapHeight = int(self.config.neuralMapHeight)
        mapWidth = int(self.config.neuralMapWidth)
        originRow = max(0, min(position[0] - mapHeight // 2, max(0, int(self.maze.height) - mapHeight)))
        originCol = max(0, min(position[1] - mapWidth // 2, max(0, int(self.maze.width) - mapWidth)))
        encoded: list[float] = []
        for row in range(originRow, min(originRow + mapHeight, int(self.maze.height))):
            for col in range(originCol, min(originCol + mapWidth, int(self.maze.width))):
                candidate = (row, col)
                knownOpen = candidate in self._knownOpen
                knownWall = candidate in self._knownWalls
                visited = self.statistics.getVisitedPositions().get(candidate, 0) > 0
                isCurrent = candidate == position
                seenGoal = candidate in self._seenGoals
                frontier = knownOpen and not visited
                encoded.extend(
                    [
                        1.0 if knownOpen else 0.0,
                        1.0 if knownWall else 0.0,
                        1.0 if visited else 0.0,
                        1.0 if isCurrent else 0.0,
                        1.0 if seenGoal else 0.0,
                        1.0 if frontier else 0.0,
                        0.0,
                    ]
                )
        return tuple(encoded)
