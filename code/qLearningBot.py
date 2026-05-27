import numpy as np
from typing import Any, cast

from botStatistics import BotStatistics
from baseBot import BaseBot
from botTools import BotTools
from botConfigs import QLearningConfig
from services.repository import ArtifactsRepository
from services.runners import QLearningEpisodeRunner


class QLearning:
    def __init__(self, qLearningConfig: QLearningConfig, repo: ArtifactsRepository | None = None, profileName: str | None = None):
        """Initialize Q-learning algorithm with the given configuration."""
        self.lr = qLearningConfig.learningRate
        self.gamma = qLearningConfig.discountFactor
        self.numActions = 4
        self.qTable: dict[tuple[Any, ...], np.ndarray[Any, Any]] = {}
        self.initialExplorationRate = 1.0
        self.minExplorationRate = 0.1
        # Decay per step across episodes (persistent)
        self.explorationDecayRate = 0.0005
        self.totalSteps = 0
        self.usePositionInState = getattr(
            qLearningConfig,
            'usePositionInState',
            getattr(qLearningConfig, 'use_position_in_state', True),
        )
        self._repo = repo
        self._profile = profileName

    def updateQValue(self, state: Any, action: int, reward: float, newState: Any) -> None:
        """ Update Q-value for the given state-action pair."""
        stateKey = self.stateToKey(state)
        newStateKey = self.stateToKey(newState)

        if stateKey not in self.qTable:
            self.qTable[stateKey] = np.zeros(self.numActions)
        if newStateKey not in self.qTable:
            self.qTable[newStateKey] = np.zeros(self.numActions)

        oldValue = self.qTable[stateKey][action]
        futureOptimalValue = np.max(self.qTable[newStateKey])
        newValue = oldValue + self.lr * (reward + self.gamma * futureOptimalValue - oldValue)
        self.qTable[stateKey][action] = newValue
    
    def chooseAction(self, state: Any, statistics: BotStatistics) -> int:
        """Choose action based on the exploration-exploitation trade-off."""
        stateKey = self.stateToKey(state)

        if stateKey not in self.qTable:
            self.qTable[stateKey] = np.zeros(self.numActions) 
        
        # Use a persistent decay across all episodes to reduce exploration over time
        explorationRate = max(
            self.minExplorationRate,
            self.initialExplorationRate - self.explorationDecayRate * self.totalSteps,
        )
        if np.random.rand() < explorationRate:
            return int(np.random.randint(self.numActions))
        return int(np.argmax(self.qTable[stateKey]))
    
    def saveQTable(self) -> None:
        if self._repo and self._profile:
            try:
                self._repo.saveQTable(self._profile, self.qTable)
            except Exception:
                pass

    def loadQTable(self) -> None:
        if self._repo and self._profile:
            try:
                self.qTable = self._repo.loadQTable(self._profile) or {}
            except Exception:
                self.qTable = {}

    def stateToKey(self, state: Any) -> tuple[Any, ...]:
        """Convert the state to a hashable key for the Q-table."""
        positionIndex, wallDistances, goalDirection = state
        if self.usePositionInState:
            return positionIndex, wallDistances, goalDirection
        # Exclude absolute position to improve generalization to new mazes
        return wallDistances, goalDirection

class QLearningBot(BaseBot):
    def __init__(
        self,
        maze: Any,
        config: QLearningConfig,
        rewardSystem: Any,
        statistics: BotStatistics,
        profileName: str,
        repository: ArtifactsRepository | None = None,
    ) -> None:
        """
        Initialize the Q-learning bot.

        :param maze: The maze object.
        :param config: Q-learning configuration.
        :param reward_system: Reward system for evaluating actions.
        :param statistics: Instance of BotStatistics for tracking statistics.
        :param profile_name: Name of the profile for saving/loading data.
        """
        cast(Any, super()).__init__(maze, statistics, config)
        self.maze: Any = maze
        self.statistics: Any = statistics
        # Use injected repository (preferred), fallback to default for backward-compat
        self.repo = repository or ArtifactsRepository()
        self.qLearning = QLearning(config, repo=self.repo, profileName=profileName)
        self.tools = BotTools(maze)
        self.rewardSystem = rewardSystem
        self.profileName = profileName
        self.totalReward = 0
        self.episodeCounter = 0
        self.position = maze.getStart()
        self.state = self.calculateState()
        self.qLearning.loadQTable()  # Load Q-table when initializing

        try:
            self.repo.ensureMazeFile(profileName)
            mazeData = self.repo.loadMazeData(profileName)
        except Exception:
            mazeData = {"highest": {"reward": float('-inf')}, "lowest": {"reward": float('inf')}}
        self.highestReward = float(mazeData.get("highest", {}).get("reward", float('-inf')))
        self.lowestReward = float(mazeData.get("lowest", {}).get("reward", float('inf')))

        # Visualization step-wise execution state
        self._visActive = False
        self._visInitialized = False
        self._visOptimalPath = None
        self._visOptimalLength = 0
        self._visStepLimit = 0
        self._visBestDistance = 0
        self._visNoProgressSteps = 0
        self._visProgressPatience = 0
        self._visSteps = 0
        self._visTimesHitWall = 0
        # Episode runner delegates training episode orchestration
        self.runner = QLearningEpisodeRunner(self)

    def getBotSpecificData(self) -> dict[str, Any]:
        """Retrieve bot-specific data."""
        return {'q_table': self.qLearning.qTable}
    
    def initializeSpecificData(self, data: dict[str, Any]) -> None:
        """Initialize bot-specific data."""
        self.qLearning.qTable = data.get('q_table', {})
        self.qLearning.loadQTable()  # Load the Q-table from a file

    def calculateState(self, position: tuple[int, int] | None = None) -> tuple[Any, ...]:
        """Calculate the state based on the given position (or current position)."""
        if position is None:
            position = cast(tuple[int, int], self.position)
        positionIndex = self.tools.posToState(position)
        wallDistances, goalDirection = self.tools.detectWalls(position)
        # visited = self.statistics.get_visited_positions()
        # distance_to_goal = self.tools.get_distance_to_goal(self.position)
        # return (position_index, wall_distances, tuple(visited), distance_to_goal, goal_direction)
        return (positionIndex, wallDistances, goalDirection)
    
    def runEpisode(self) -> None:
        """Run a single episode of Q-learning (delegated to episode runner)."""
        self.runner.runEpisode()

    def reset(self) -> None:
        """Reset the bot's position, statistics, and Q-learning data."""
        self.position = self.maze.start
        self.statistics.reset()
        self.totalReward = 0
        self.state = self.calculateState()

    # Training lifecycle hooks (explicit contract implementation)
    def onEpisodeStart(self, mode: str) -> None:
        if mode != "training":
            raise ValueError(f"Unsupported episode mode for QLearningBot: {mode}")

    def onEpisodeStep(self, mode: str, stepIndex: int) -> None:
        if mode != "training":
            raise ValueError(f"Unsupported episode mode for QLearningBot: {mode}")
        if stepIndex < 0:
            raise ValueError(f"step_index must be non-negative, got {stepIndex}")

    def onEpisodeEnd(self, mode: str, outcome: str) -> None:
        if mode != "training":
            raise ValueError(f"Unsupported episode mode for QLearningBot: {mode}")
        if not outcome:
            raise ValueError("outcome must be a non-empty string")

    # ---- Visualization step-wise execution helpers ----
    def beginVisualizationEpisode(self) -> None:
        """Initialize state for a step-wise episode run used by visualization."""
        optimalPath = self.tools.getOptimalPathInfo(self.maze.start, self.maze.end, output='path')
        if isinstance(optimalPath, int):
            optimalPath = []
        optimalLength = len(optimalPath)
        areaBonus = int(0.5 * self.maze.width * self.maze.height)
        stepLimit = min(5000, max(200, 12 * optimalLength + areaBonus)) if optimalLength > 0 else max(200, areaBonus)

        def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        self.position = self.maze.getStart()
        self.statistics.reset()
        self.totalReward = 0
        self.state = self.calculateState()

        self._visActive = True
        self._visInitialized = True
        self._visOptimalPath = optimalPath
        self._visOptimalLength = optimalLength
        self._visStepLimit = stepLimit
        self._visBestDistance = manhattan(cast(tuple[int, int], self.position), cast(tuple[int, int], self.maze.end))
        self._visNoProgressSteps = 0
        self._visProgressPatience = min(200, 50 * optimalLength)
        self._visSteps = 0
        self._visTimesHitWall = 0

    def stepVisualization(self, maxSteps: int = 1) -> bool:
        """
        Perform up to max_steps steps for live visualization only.

        Unified visualization semantics (singular policy across bots):
        - No learning/backprop or Q-table updates during visualization
        - No persistence of rewards, heatmaps, or profile counters
        - Only in-memory bot position and statistics.heatmap are updated for drawing
        - Returns True when the episode finishes; caller resets environment
        """
        if not self._visInitialized:
            self.beginVisualizationEpisode()

        for _ in range(maxSteps):
            if self.position == self.maze.end:
                return self._finalizeVisualizationEpisode()

            action = self.qLearning.chooseAction(self.state, self.statistics)
            newPosition = self.tools.calculateNextPosition(self.position, action)
            self.statistics.totalSteps = self.statistics.timesRevisitedSquares + self.statistics.nonRepeatingStepsTaken

            reward = 0
            if not self.maze.isValidPosition(self.profileName, newPosition[0], newPosition[1]):
                # In visualization: compute reward for display consistency, but do not learn
                reward += self.rewardSystem.getReward(self.position, newPosition, self._visOptimalPath, self._visOptimalLength, self.statistics.getVisitedPositions())
                # Do not update Q-values or global training step counters here
                self.totalReward += reward
                self._visTimesHitWall += 1
                # continue to next step without moving
            else:
                # Track the previous position, then count the new cell after moving
                self.statistics.updateLastVisited(self.position)
                reward += self.rewardSystem.getReward(self.position, newPosition, self._visOptimalPath, self._visOptimalLength, self.statistics.getVisitedPositions())

                if newPosition in self.statistics.getVisitedPositions():
                    self.statistics.timesRevisitedSquares += 1
                else:
                    self.statistics.nonRepeatingStepsTaken += 1

                if self.statistics.totalSteps > self._visStepLimit:
                    reward += -100
                    self.totalReward += reward
                    return self._finalizeVisualizationEpisode()

                self.totalReward += reward
                # Advance environment state only; do not learn
                newState = self.calculateState(newPosition)
                self.position = newPosition
                self.statistics.updateVisitedPositions(self.position)
                self.state = newState
                self._visSteps += 1

                # Progress tracking for early-stop (currently not used to break early)
                def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
                    return abs(a[0]-b[0]) + abs(a[1]-b[1])
                currentDistance = manhattan(self.position, self.maze.end)
                if currentDistance < self._visBestDistance:
                    self._visBestDistance = currentDistance
                    self._visNoProgressSteps = 0
                else:
                    self._visNoProgressSteps += 1

            if self.position == self.maze.end:
                return self._finalizeVisualizationEpisode()

        return False

    def _finalizeVisualizationEpisode(self) -> bool:
        """
        Finalize a visualization-only episode.

        Unified policy: do NOT persist any training artifacts here. Leave
        Q-table, rewards log, and profile stats untouched. Only reset local
        visualization state so a fresh episode can start on next call.
        """
        self._visActive = False
        self._visInitialized = False
        return True
