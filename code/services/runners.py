from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeRuntime:
    optimalPath: list[tuple[int, int]]
    optimalLength: int
    stepLimit: int
    steps: int = 0
    timesHitWall: int = 0
    outcome: str = "aborted"
    bestDistance: int = 0


class QLearningEpisodeRunner:
    """
    Episode controller for QLearningBot (training episodes only).

    Contract:
    - Owns episode orchestration (start/step/end loop control).
    - Delegates algorithm internals to the bot (policy/value updates).
    - Triggers persistence through repository calls at episode end.
    - Must not be used for visualization; visualization remains inference-only.
    """

    def __init__(self, bot: Any):
        self.bot = bot

    def runEpisode(self) -> None:
        bot = self.bot
        bot.onEpisodeStart("training")
        runtime = self._startEpisode()
        try:
            while bot.position != bot.maze.end:
                if not self._runStep(runtime):
                    break
            if bot.position == bot.maze.end:
                runtime.outcome = "goal_reached"
            self._finalizeEpisode(runtime)
        finally:
            bot.onEpisodeEnd("training", runtime.outcome)

    def _startEpisode(self) -> EpisodeRuntime:
        bot = self.bot
        tools = bot.tools
        maze = bot.maze

        optimalPath = tools.getOptimalPathInfo(maze.start, maze.end, output='path')
        optimalLength = len(optimalPath)
        areaBonus = int(0.5 * maze.width * maze.height)
        stepLimit = min(5000, max(200, 12 * optimalLength + areaBonus)) if optimalLength > 0 else max(200, areaBonus)

        bot.totalReward = 0
        return EpisodeRuntime(
            optimalPath=optimalPath,
            optimalLength=optimalLength,
            stepLimit=stepLimit,
            bestDistance=self._manhattan(bot.position, maze.end),
        )

    def _runStep(self, runtime: EpisodeRuntime) -> bool:
        bot = self.bot
        tools = bot.tools
        maze = bot.maze
        stats = bot.statistics
        rsys = bot.rewardSystem
        ql = bot.qLearning

        bot.onEpisodeStep("training", runtime.steps)
        reward = 0
        action = ql.chooseAction(bot.state, stats)
        newPosition = tools.calculateNextPosition(bot.position, action)
        stats.totalSteps = stats.timesRevisitedSquares + stats.nonRepeatingStepsTaken

        if not maze.isValidPosition(bot.profileName, newPosition[0], newPosition[1]):
            reward += rsys.getReward(bot.position, newPosition, runtime.optimalPath, runtime.optimalLength, stats.getVisitedPositions())
            newState = bot.calculateState()
            ql.updateQValue(bot.state, action, reward, newState)
            bot.totalReward += reward
            runtime.timesHitWall += 1
            ql.totalSteps += 1
            return True

        stats.updateLastVisited(bot.position)
        reward += rsys.getReward(bot.position, newPosition, runtime.optimalPath, runtime.optimalLength, stats.getVisitedPositions())

        if newPosition in stats.getVisitedPositions():
            stats.timesRevisitedSquares += 1
        else:
            stats.nonRepeatingStepsTaken += 1

        if stats.totalSteps > runtime.stepLimit:
            reward += -100
            newState = bot.calculateState()
            ql.updateQValue(bot.state, action, reward, newState)
            bot.totalReward += reward
            runtime.outcome = "step_limit_statistics"
            return False

        bot.totalReward += reward
        newState = bot.calculateState(newPosition)
        ql.updateQValue(bot.state, action, reward, newState)
        ql.totalSteps += 1

        bot.position = newPosition
        stats.updateVisitedPositions(bot.position)
        bot.state = newState
        runtime.steps += 1

        currentDistance = self._manhattan(bot.position, maze.end)
        if currentDistance < runtime.bestDistance:
            runtime.bestDistance = currentDistance
        if runtime.steps > runtime.stepLimit:
            runtime.outcome = "step_limit_loop"
            return False
        return True

    def _finalizeEpisode(self, runtime: EpisodeRuntime) -> None:
        bot = self.bot
        maze = bot.maze
        heatmapData = bot.statistics.getVisitedPositions()
        try:
            bot.repo.saveMazeEpisode(bot.profileName, maze, heatmapData, bot.totalReward)
            bot.repo.updateStepsFromHeatmap(bot.profileName, heatmapData)
            if runtime.timesHitWall:
                bot.repo.incrementTimesHitWall(bot.profileName, runtime.timesHitWall)
        except Exception:
            pass
        bot.repo.appendReward(bot.profileName, bot.totalReward)
        bot.episodeCounter += 1
        bot.qLearning.saveQTable()

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

