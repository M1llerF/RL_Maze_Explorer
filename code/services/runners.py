from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from rewardSystem import clipRewardValue
from rewardSystem import RewardEvent
from environment.context import ENEMY_KILLED
from environment.enemy_system import EnemySystem
from services.episode_result import EpisodeResult
from bots.common.actions import NUM_PRIMITIVE_DIRECTIONS


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

    Runner responsibilities:
    - Episode loop: action selection, step execution, termination checks.
    - Bot lifecycle hooks: onEpisodeStart/Step/End.
    - Returns EpisodeResult for the recorder to handle persistence.

    Learning updates (Q-value, SMDP) are delegated to bot.applyLearningUpdate().
    Persistence (repo writes, Q-table save) is handled by PostEpisodeRecorder.
    """

    def __init__(self, bot: Any):
        self.bot = bot
        self._visualizationRuntime: EpisodeRuntime | None = None
        self._enemySystem = EnemySystem()
        self._pendingKills: int = 0
        self._episodeKills: int = 0
        context = getattr(bot, "_context", None)
        if context is not None:
            context.on(ENEMY_KILLED, lambda **_: self._onEnemyKilled())

    def runEpisode(self) -> EpisodeResult:
        bot = self.bot
        bot.onEpisodeStart("training")
        runtime = self._startEpisode(mode="training")
        try:
            while bot.position != bot.maze.end:
                if not self._runStep(runtime, mode="training", apply_learning=True):
                    break
            if bot.position == bot.maze.end:
                runtime.outcome = "goal_reached"
            return self._finalizeEpisode(runtime)
        finally:
            bot.onEpisodeEnd("training", runtime.outcome)

    def beginVisualizationEpisode(self) -> None:
        bot = self.bot
        bot.onEpisodeStart("visualization")
        self._visualizationRuntime = self._startEpisode(mode="visualization")

    def stepVisualization(self, maxSteps: int = 1) -> bool:
        if self._visualizationRuntime is None:
            self.beginVisualizationEpisode()
        runtime = self._visualizationRuntime
        if runtime is None:
            return True
        bot = self.bot
        for _ in range(max(1, int(maxSteps))):
            if bot.position == bot.maze.end:
                runtime.outcome = "goal_reached"
                return self._finalizeVisualizationEpisode(runtime)
            if not self._runStep(runtime, mode="visualization", apply_learning=False):
                if bot.position == bot.maze.end and runtime.outcome == "aborted":
                    runtime.outcome = "goal_reached"
                return self._finalizeVisualizationEpisode(runtime)
        return False

    def _startEpisode(self, *, mode: str) -> EpisodeRuntime:
        bot = self.bot
        tools = bot.tools
        maze = bot.maze

        if mode == "visualization":
            bot.reset()

        optimalPath = tools.getOptimalPathInfo(maze.start, maze.end, output="path")
        optimalLength = len(optimalPath)
        areaBonus = int(0.5 * maze.width * maze.height)
        stepLimit = min(5000, max(200, 12 * optimalLength + areaBonus)) if optimalLength > 0 else max(200, areaBonus)

        bot.totalReward = 0.0
        bot.currentEpisodeSteps = 0
        self._pendingKills = 0
        self._episodeKills = 0
        bot._currentOptimalPath = optimalPath
        bot._currentOptimalLength = optimalLength
        if mode == "visualization":
            bot._visActive = True
            bot._visInitialized = True
            bot._visOptimalPath = optimalPath
            bot._visOptimalLength = optimalLength
            bot._visStepLimit = stepLimit
            bot._visBestDistance = self._manhattan(bot.position, maze.end)
            bot._visNoProgressSteps = 0
            bot._visProgressPatience = min(200, 50 * optimalLength)
            bot._visSteps = 0
            bot._visTimesHitWall = 0
        return EpisodeRuntime(
            optimalPath=optimalPath,
            optimalLength=optimalLength,
            stepLimit=stepLimit,
            bestDistance=self._manhattan(bot.position, maze.end),
        )

    def _onEnemyKilled(self) -> None:
        self._pendingKills += 1
        self._episodeKills += 1

    def _collectKillEvents(self) -> tuple[RewardEvent, ...]:
        kills, self._pendingKills = self._pendingKills, 0
        return tuple(RewardEvent(name="enemy_killed", payload={}) for _ in range(kills))

    def _tickEnemies(self) -> Any:
        context = getattr(self.bot, "_context", None)
        if context is None:
            return None
        return self._enemySystem.tick(context)

    @staticmethod
    def _rewardEventsFromEnemyTick(enemyTick: Any) -> tuple[RewardEvent, ...]:
        if enemyTick is None:
            return ()
        events = [
            RewardEvent(name=str(item.get("name", "")), payload=dict(item.get("payload", {})))
            for item in tuple(getattr(enemyTick, "semantic_events", ()) or ())
            if isinstance(item, dict)
        ]
        if bool(getattr(enemyTick, "agent_caught", False)):
            events.append(RewardEvent(name="death_by_enemy", payload={}))
        return tuple(events)

    def _runStep(self, runtime: EpisodeRuntime, *, mode: str, apply_learning: bool) -> bool:
        bot = self.bot
        stats = bot.statistics
        rsys = bot.rewardSystem
        maze = bot.maze

        bot.onEpisodeStep(mode, runtime.steps)
        stats.totalSteps = stats.timesRevisitedSquares + stats.nonRepeatingStepsTaken

        decision = bot.buildDecisionInput()
        choice = bot.selectAction(decision)
        start_state = bot.state
        semanticAction = decision.actionSpace.semanticId(choice.local_id)
        is_option = bot.isOptionAction(semanticAction)

        if is_option:
            result = bot.applyStep(choice.local_id)
            enemyTick = self._tickEnemies()
            rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
            reward = float(result.reward)
            if hasattr(rsys, "evaluateSemanticEvents"):
                reward += float(rsys.evaluateSemanticEvents(rewardEvents))
            reward = clipRewardValue(float(reward), bot.config)
            bot.totalReward += reward
            if apply_learning:
                bot.applyLearningUpdate(start_state, choice.local_id, reward, int(result.duration))
            runtime.steps += result.duration
            bot.currentEpisodeSteps = int(runtime.steps)
            if mode == "visualization":
                bot._visSteps = int(runtime.steps)
            if bool(getattr(enemyTick, "agent_caught", False)):
                runtime.outcome = "death_by_enemy"
                return False
            if result.done or bot.position == maze.end:
                runtime.outcome = "goal_reached" if bot.position == maze.end else runtime.outcome
                return False
            if runtime.steps > runtime.stepLimit:
                runtime.outcome = "step_limit_loop"
                return False
            return True

        # Attack primitives (semantic IDs >= NUM_PRIMITIVE_DIRECTIONS) don't move the bot.
        # Execute them directly without the movement-validity path.
        if semanticAction >= NUM_PRIMITIVE_DIRECTIONS:
            result = bot.executePrimitiveAction(semanticAction)
            enemyTick = self._tickEnemies()
            rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
            reward = float(result.reward)
            if hasattr(rsys, "evaluateSemanticEvents"):
                reward += float(rsys.evaluateSemanticEvents(rewardEvents))
            reward = clipRewardValue(reward, bot.config)
            bot.totalReward += reward
            if apply_learning:
                bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)
            if bool(getattr(enemyTick, "agent_caught", False)):
                runtime.outcome = "death_by_enemy"
                return False
            runtime.steps += 1
            bot.currentEpisodeSteps = int(runtime.steps)
            if mode == "visualization":
                bot._visSteps = int(runtime.steps)
            if runtime.steps > runtime.stepLimit:
                runtime.outcome = "step_limit_loop"
                return False
            return True

        attempted = bot.tools.calculateNextPosition(bot.position, semanticAction)
        context = getattr(bot, "_context", None)
        is_valid_attempt = (
            bool(context.isValidPosition(attempted))
            if context is not None
            else bool(maze.isValidPosition(bot.profileName, attempted[0], attempted[1]))
        )
        rewardContext = rsys.buildStepContext(
            bot.position,
            attempted,
            runtime.optimalPath,
            runtime.optimalLength,
            stats.getVisitedPositions(),
        ) if hasattr(rsys, "buildStepContext") else None
        reward = float(
            rsys.evaluateStep(rewardContext)
            if rewardContext is not None and hasattr(rsys, "evaluateStep")
            else rsys.getReward(
                bot.position, attempted, runtime.optimalPath, runtime.optimalLength, stats.getVisitedPositions()
            )
        )

        if not is_valid_attempt:
            reward = clipRewardValue(float(reward), bot.config)
            if apply_learning:
                bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)
            bot.totalReward += reward
            runtime.timesHitWall += 1
            if mode == "visualization":
                bot._visTimesHitWall = int(runtime.timesHitWall)
            bot.currentEpisodeSteps = int(runtime.steps)
            return True

        result = bot.executePrimitiveAction(semanticAction)
        if result.info.get("death_by_enemy"):
            reward = clipRewardValue(float(reward + float(result.reward)), bot.config)
            bot.totalReward += reward
            if apply_learning:
                bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)
            runtime.outcome = "death_by_enemy"
            return False
        reward += result.reward
        enemyTick = self._tickEnemies()
        rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
        if hasattr(rsys, "evaluateSemanticEvents"):
            reward += float(rsys.evaluateSemanticEvents(rewardEvents))

        stats.totalSteps = stats.timesRevisitedSquares + stats.nonRepeatingStepsTaken

        if stats.totalSteps > runtime.stepLimit:
            reward += -100.0
            reward = clipRewardValue(float(reward), bot.config)
            if apply_learning:
                bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)
            bot.totalReward += reward
            runtime.outcome = "step_limit_statistics"
            return False

        reward = clipRewardValue(float(reward), bot.config)
        bot.totalReward += reward
        if apply_learning:
            bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)

        if bool(getattr(enemyTick, "agent_caught", False)):
            runtime.outcome = "death_by_enemy"
            return False

        runtime.steps += 1
        bot.currentEpisodeSteps = int(runtime.steps)
        if mode == "visualization":
            bot._visSteps = int(runtime.steps)

        currentDistance = self._manhattan(bot.position, maze.end)
        if currentDistance < runtime.bestDistance:
            runtime.bestDistance = currentDistance
        if runtime.steps > runtime.stepLimit:
            runtime.outcome = "step_limit_loop"
            return False
        return True

    def _finalizeVisualizationEpisode(self, runtime: EpisodeRuntime) -> bool:
        bot = self.bot
        bot.onEpisodeEnd("visualization", runtime.outcome)
        bot._visActive = False
        bot._visInitialized = False
        bot._currentOptimalPath = []
        bot._currentOptimalLength = 0
        bot._visOutcome = str(runtime.outcome)
        bot._visWallHits = int(runtime.timesHitWall)
        bot._visStepCount = int(runtime.steps)
        bot._visEnemyKills = 0
        self._visualizationRuntime = None
        return True

    def _finalizeEpisode(self, runtime: EpisodeRuntime) -> EpisodeResult:
        bot = self.bot
        bot.lastEpisodeSuccess = runtime.outcome == "goal_reached"
        bot.lastEpisodeSteps = int(runtime.steps)
        bot.lastEpisodeOptimalSteps = int(runtime.optimalLength)
        bot.lastEpisodeOutcome = str(runtime.outcome)
        bot.lastEpisodeWallHits = int(runtime.timesHitWall)
        bot.lastEpisodeEnemyKills = self._episodeKills
        bot.episodeCounter += 1
        bot._currentOptimalPath = []
        bot._currentOptimalLength = 0
        heatmapData = dict(bot.statistics.getVisitedPositions())
        return EpisodeResult(
            profile_name=str(bot.profileName),
            mode="training",
            outcome=runtime.outcome,
            success=runtime.outcome == "goal_reached",
            total_reward=float(bot.totalReward),
            steps=int(runtime.steps),
            optimal_steps=int(runtime.optimalLength),
            times_hit_wall=int(runtime.timesHitWall),
            heatmap_data=heatmapData,
            maze=bot.maze,
            save_maze_episode=True,
            save_heatmap_stats=True,
            append_reward=True,
        )

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
        self._enemySystem = EnemySystem()
        self._pendingKills: int = 0
        self._episodeKills: int = 0
        context = getattr(bot, "_context", None)
        if context is not None:
            context.on(ENEMY_KILLED, lambda **_: self._onEnemyKilled())

    def runEpisode(self, mode: str = "training") -> EpisodeResult:
        bot = self.bot
        if mode not in {"training", "evaluation"}:
            raise ValueError(f"Unsupported episode mode: {mode}")
        bot.onEpisodeStart(mode)
        runtime = self._startEpisode()
        try:
            while bot.position != bot.maze.end:
                if bot.isStopRequested():
                    runtime.outcome = "stop_requested"
                    break
                if not self._runStep(runtime, mode=mode):
                    break
            if bot.position == bot.maze.end:
                runtime.outcome = "goal_reached"
            return self._finalizeEpisode(runtime, mode=mode)
        finally:
            bot.onEpisodeEnd(mode, runtime.outcome)

    def _startEpisode(self) -> DQNEpisodeRuntime:
        bot = self.bot
        optimalPathRaw = bot.tools.getOptimalPathInfo(bot.maze.start, bot.maze.end, output="path")
        optimalPath: list[tuple[int, int]] = cast(list[tuple[int, int]], optimalPathRaw) if isinstance(optimalPathRaw, list) else []
        optimalLength = len(optimalPath)
        cfg = bot.config
        areaBonus = int(float(cfg.stepLimitAreaCoeff) * bot.maze.width * bot.maze.height)
        dynamicLimit = (
            min(int(cfg.stepLimitMax), max(int(cfg.stepLimitMin), int(cfg.stepLimitStepCoeff) * optimalLength + areaBonus))
            if optimalLength > 0
            else max(int(cfg.stepLimitMin), areaBonus)
        )
        stepLimit = min(dynamicLimit, int(cfg.maxStepsPerEpisode))
        dynamicPatience = int(max(1, optimalLength) * float(cfg.noProgressPatienceFactor))
        progressPatience = min(
            stepLimit,
            max(int(cfg.minNoProgressSteps), min(int(cfg.maxNoProgressSteps), dynamicPatience)),
        )
        bot.reset()
        bot.currentEpisodeSteps = 0
        self._pendingKills = 0
        self._episodeKills = 0
        bot._currentOptimalPath = optimalPath
        bot._currentOptimalLength = optimalLength
        return DQNEpisodeRuntime(
            optimalPath=optimalPath,
            optimalLength=optimalLength,
            stepLimit=stepLimit,
            progressPatience=progressPatience,
        )

    def _onEnemyKilled(self) -> None:
        self._pendingKills += 1
        self._episodeKills += 1

    def _collectKillEvents(self) -> tuple[RewardEvent, ...]:
        kills, self._pendingKills = self._pendingKills, 0
        return tuple(RewardEvent(name="enemy_killed", payload={}) for _ in range(kills))

    def _tickEnemies(self) -> Any:
        context = getattr(self.bot, "_context", None)
        if context is None:
            return None
        return self._enemySystem.tick(context)

    @staticmethod
    def _rewardEventsFromEnemyTick(enemyTick: Any) -> tuple[RewardEvent, ...]:
        if enemyTick is None:
            return ()
        events = [
            RewardEvent(name=str(item.get("name", "")), payload=dict(item.get("payload", {})))
            for item in tuple(getattr(enemyTick, "semantic_events", ()) or ())
            if isinstance(item, dict)
        ]
        if bool(getattr(enemyTick, "agent_caught", False)):
            events.append(RewardEvent(name="death_by_enemy", payload={}))
        return tuple(events)

    def _runStep(self, runtime: DQNEpisodeRuntime, *, mode: str) -> bool:
        if bool(getattr(self.bot, "usesHierarchicalPolicy", False)):
            return self._runHierarchicalStep(runtime, mode=mode)
        bot = self.bot
        stats = bot.statistics
        stats.totalSteps = stats.timesRevisitedSquares + stats.nonRepeatingStepsTaken
        bot.onEpisodeStep(mode, runtime.steps)
        decision = bot.buildDecisionInput(training=True)
        encodedState = decision.state
        choice = (
            bot.chooseAction(training=True)
            if mode == "training"
            else bot.chooseAction(
                training=False,
                forceEpsilon=float(bot.config.epsilonEnd),
                allowWarmupPlanner=False,
            )
        )

        result = bot.applyStep(choice.local_id, training=(mode == "training"))
        reward = float(result.reward)
        if result.info.get("death_by_enemy"):
            runtime.outcome = "death_by_enemy"
        enemyTick = self._tickEnemies()
        rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
        if hasattr(bot.rewardSystem, "evaluateSemanticEvents"):
            reward += float(bot.rewardSystem.evaluateSemanticEvents(rewardEvents))
        hitWall = bool(result.info.get("hit_wall", False))
        wasRevisit = bool(result.info.get("was_revisit", False))
        done = bool(result.done or bot.position == bot.maze.end)
        if hitWall:
            runtime.timesHitWall += 1
        else:
            if wasRevisit:
                runtime.noProgressSteps += 1
            else:
                runtime.noProgressSteps = 0
            if bot.position == bot.maze.end:
                done = True

        runtime.steps += int(result.duration)
        bot.currentEpisodeSteps = int(runtime.steps)

        if not done and runtime.steps >= runtime.stepLimit:
            reward += float(bot.config.stepLimitPenalty)
            runtime.outcome = "step_limit"
            done = True
        if not done and bool(getattr(enemyTick, "agent_caught", False)):
            runtime.outcome = "death_by_enemy"
            done = True
        if not done and runtime.noProgressSteps >= runtime.progressPatience:
            reward += float(bot.config.noProgressPenalty)
            runtime.outcome = "no_progress"
            done = True
        reward = clipRewardValue(float(reward), bot.config)

        if mode == "training":
            bot.applyLearningUpdate(encodedState, choice.local_id, reward, done, int(result.duration))
        bot.addReward(reward)
        return not done

    def _runHierarchicalStep(self, runtime: DQNEpisodeRuntime, *, mode: str) -> bool:
        bot = self.bot
        stats = bot.statistics
        stats.totalSteps = stats.timesRevisitedSquares + stats.nonRepeatingStepsTaken
        bot.onEpisodeStep(mode, runtime.steps)
        decision = bot.buildHighLevelDecisionInput()
        encodedState = decision.state
        choice = (
            bot.chooseAction(training=True)
            if mode == "training"
            else bot.chooseAction(
                training=False,
                forceEpsilon=float(bot.config.epsilonEnd),
                allowWarmupPlanner=False,
            )
        )
        result = bot.executeHierarchicalOption(choice.local_id, training=(mode == "training"))
        reward = float(result.reward)
        if result.info.get("death_by_enemy"):
            runtime.outcome = "death_by_enemy"
        enemyTick = self._tickEnemies()
        rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
        if hasattr(bot.rewardSystem, "evaluateSemanticEvents"):
            reward += float(bot.rewardSystem.evaluateSemanticEvents(rewardEvents))
        hitWall = bool(result.info.get("hit_wall", False))
        wasRevisit = bool(result.info.get("was_revisit", False))
        done = bool(result.done or bot.position == bot.maze.end)

        if hitWall:
            runtime.timesHitWall += 1
        else:
            runtime.noProgressSteps = runtime.noProgressSteps + 1 if wasRevisit else 0
            if bot.position == bot.maze.end:
                done = True

        runtime.steps += int(result.duration)
        bot.currentEpisodeSteps = int(runtime.steps)

        if not done and runtime.steps >= runtime.stepLimit:
            reward += float(bot.config.stepLimitPenalty)
            runtime.outcome = "step_limit"
            done = True
        if not done and bool(getattr(enemyTick, "agent_caught", False)):
            runtime.outcome = "death_by_enemy"
            done = True
        if not done and runtime.noProgressSteps >= runtime.progressPatience:
            reward += float(bot.config.noProgressPenalty)
            runtime.outcome = "no_progress"
            done = True
        reward = clipRewardValue(float(reward), bot.config)

        if mode == "training":
            bot.applyHighLevelLearningUpdate(encodedState, choice.local_id, reward, done, int(result.duration))
        bot.addReward(reward)
        return not done

    def _finalizeEpisode(self, runtime: DQNEpisodeRuntime, *, mode: str) -> EpisodeResult:
        bot = self.bot
        success = runtime.outcome == "goal_reached"
        is_warming_up = mode == "training" and bool(bot.isWarmingUp)

        if mode == "training":
            bot.lastEpisodeSuccess = success
            bot.lastEpisodeSteps = int(runtime.steps)
            bot.lastEpisodeOptimalSteps = int(runtime.optimalLength)
            bot.lastEpisodeOutcome = str(runtime.outcome)
            bot.lastEpisodeWallHits = int(runtime.timesHitWall)
            bot.lastEpisodeEnemyKills = self._episodeKills
        else:
            bot.lastEvaluationSuccess = success
            bot.lastEvaluationSteps = int(runtime.steps)
            bot.lastEvaluationOptimalSteps = int(runtime.optimalLength)
            bot.lastEvaluationReward = float(bot.totalReward)

        heatmapData = dict(bot.statistics.getVisitedPositions())
        if not heatmapData:
            heatmapData[tuple(bot.position)] = 1

        bot.episodeCounter += 1 if mode == "training" else 0
        bot.currentEpisodeSteps = 0
        bot._currentOptimalPath = []
        bot._currentOptimalLength = 0

        if is_warming_up:
            return EpisodeResult(
                profile_name=str(bot.profileName),
                mode=mode,
                outcome=runtime.outcome,
                success=False,
                total_reward=float(bot.totalReward),
                steps=int(runtime.steps),
                optimal_steps=int(runtime.optimalLength),
                times_hit_wall=0,
                heatmap_data={},
                maze=bot.maze,
                save_maze_episode=False,
                save_heatmap_stats=False,
                append_reward=False,
                is_warming_up=True,
            )

        return EpisodeResult(
            profile_name=str(bot.profileName),
            mode=mode,
            outcome=runtime.outcome,
            success=success,
            total_reward=float(bot.totalReward),
            steps=int(runtime.steps),
            optimal_steps=int(runtime.optimalLength),
            times_hit_wall=int(runtime.timesHitWall),
            heatmap_data=heatmapData,
            maze=bot.maze,
            save_maze_episode=(mode == "evaluation"),
            save_heatmap_stats=(mode == "training"),
            append_reward=(mode == "evaluation"),
            is_warming_up=False,
        )
