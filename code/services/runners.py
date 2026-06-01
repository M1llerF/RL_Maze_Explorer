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
        bot.currentEpisodeSteps = 0
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
            bot.currentEpisodeSteps = int(runtime.steps)
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
        bot.currentEpisodeSteps = int(runtime.steps)

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


@dataclass
class DQNEpisodeRuntime:
    optimalPath: list[tuple[int, int]]
    optimalLength: int
    stepLimit: int
    progressPatience: int
    steps: int = 0
    timesHitWall: int = 0
    noProgressSteps: int = 0
    outcome: str = "aborted"


class DQNEpisodeRunner:
    def __init__(self, bot: Any):
        self.bot = bot

    def runEpisode(self) -> None:
        bot = self.bot
        bot.onEpisodeStart("training")
        runtime = self._startEpisode()
        try:
            while bot.position != bot.maze.end:
                if bot._stopRequested:
                    runtime.outcome = "stop_requested"
                    break
                if not self._runStep(runtime):
                    break
            if bot.position == bot.maze.end:
                runtime.outcome = "goal_reached"
            self._finalizeEpisode(runtime)
        finally:
            bot.onEpisodeEnd("training", runtime.outcome)

    def _startEpisode(self) -> DQNEpisodeRuntime:
        bot = self.bot
        optimalPathRaw = bot.tools.getOptimalPathInfo(bot.maze.start, bot.maze.end, output="path")
        optimalPath = optimalPathRaw if isinstance(optimalPathRaw, list) else []
        optimalLength = len(optimalPath)
        areaBonus = int(0.5 * bot.maze.width * bot.maze.height)
        dynamicLimit = min(5000, max(200, 12 * optimalLength + areaBonus)) if optimalLength > 0 else max(200, areaBonus)
        stepLimit = min(dynamicLimit, int(bot.config.maxStepsPerEpisode))
        dynamicPatience = int(max(1, optimalLength) * float(bot.config.noProgressPatienceFactor))
        progressPatience = min(
            stepLimit,
            max(int(bot.config.minNoProgressSteps), min(int(bot.config.maxNoProgressSteps), dynamicPatience)),
        )
        bot.reset()
        bot.currentEpisodeSteps = 0
        return DQNEpisodeRuntime(
            optimalPath=optimalPath,
            optimalLength=optimalLength,
            stepLimit=stepLimit,
            progressPatience=progressPatience,
        )

    def _runStep(self, runtime: DQNEpisodeRuntime) -> bool:
        bot = self.bot
        stats = bot.statistics
        stats.totalSteps = stats.timesRevisitedSquares + stats.nonRepeatingStepsTaken
        bot.onEpisodeStep("training", runtime.steps)
        encodedState = bot.encodeState()
        action = bot.chooseAction(encodedState, training=True)
        attempted = bot.tools.calculateNextPosition(bot.position, action)
        reward = float(
            bot.rewardSystem.getReward(
                bot.position,
                attempted,
                runtime.optimalPath,
                runtime.optimalLength,
                stats.getVisitedPositions(),
            )
        )
        done = False
        if not bot.maze.isValidPosition(bot.profileName, attempted[0], attempted[1]):
            runtime.timesHitWall += 1
            runtime.steps += 1
            bot.currentEpisodeSteps = int(runtime.steps)
            bot._rememberWall(attempted)
            nextStateRaw = bot.calculateState()
        else:
            stats.updateLastVisited(bot.position)
            if attempted in stats.getVisitedPositions():
                stats.timesRevisitedSquares += 1
                reward -= float(bot.config.repeatVisitPenaltyScale)
            else:
                stats.nonRepeatingStepsTaken += 1
                runtime.noProgressSteps = 0
            if bot.previousPosition is not None and attempted == bot.previousPosition:
                reward += float(bot.config.immediateReversalPenalty)
            bot.previousPosition = bot.position
            bot.position = attempted
            stats.updateVisitedPositions(bot.position)
            bot._observePosition(bot.position)
            runtime.steps += 1
            bot.currentEpisodeSteps = int(runtime.steps)
            runtime.noProgressSteps += 1
            nextStateRaw = bot.calculateState()
            if bot.position == bot.maze.end:
                done = True

        if not done and runtime.steps >= runtime.stepLimit:
            reward += -100.0
            runtime.outcome = "step_limit"
            done = True
        if not done and runtime.noProgressSteps >= runtime.progressPatience:
            reward += float(bot.config.noProgressPenalty)
            runtime.outcome = "no_progress"
            done = True

        nextEncoded = bot.encodeState(nextStateRaw)
        transition = bot.makeTransition(encodedState, action, reward, nextEncoded, done)
        bot.agent.storeTransition(transition)
        bot.agent.trainStep()
        bot.agent.onEnvironmentStep()
        bot.state = nextStateRaw
        bot.totalReward += reward
        return not done

    def _finalizeEpisode(self, runtime: DQNEpisodeRuntime) -> None:
        bot = self.bot
        heatmapData = dict(bot.statistics.getVisitedPositions())
        if not heatmapData:
            heatmapData[tuple(bot.position)] = 1
        try:
            bot.repo.saveMazeEpisode(bot.profileName, bot.maze, heatmapData, bot.totalReward)
            bot.repo.updateStepsFromHeatmap(bot.profileName, heatmapData)
            if runtime.timesHitWall:
                bot.repo.incrementTimesHitWall(bot.profileName, runtime.timesHitWall)
        except Exception:
            pass
        try:
            bot.repo.appendReward(bot.profileName, bot.totalReward)
        except Exception:
            pass
        bot.episodeCounter += 1
        bot.currentEpisodeSteps = 0
        bot.saveCheckpoint()
