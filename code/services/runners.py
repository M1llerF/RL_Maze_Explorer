from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable
from rewardSystem import clipRewardValue
from rewardSystem import RewardEvent
from environment.context import ENEMY_KILLED
from environment.enemySystem import EnemySystem
from services.episodePolicy import compute_no_progress_patience, compute_step_limit
from services.episodeResult import EpisodePersistencePolicy, EpisodeResult, EvaluationEpisodeDefinition
from bots.common.actions import NUM_PRIMITIVE_DIRECTIONS


@runtime_checkable
class EpisodeRunnerProtocol(Protocol):
    """Common interface for all episode runners used by the research system."""

    def runEpisode(self, *, mode: str = "training") -> EpisodeResult: ...
    def runEvaluationEpisode(self) -> EpisodeResult: ...


def _resolve_episode_persistence_policy(
    bot: Any,
    mode: str,
    *,
    fallback: EpisodePersistencePolicy,
) -> EpisodePersistencePolicy:
    getter = getattr(bot, "getEpisodePersistencePolicy", None)
    if not callable(getter):
        return fallback
    policy = getter(mode)
    if isinstance(policy, EpisodePersistencePolicy):
        return policy
    return fallback


def _resolve_evaluation_epsilon(bot: Any, fallback: float | None = None) -> float | None:
    getter = getattr(bot, "getEvaluationEpisodeDefinition", None)
    if not callable(getter):
        return fallback
    definition = getter()
    if isinstance(definition, EvaluationEpisodeDefinition):
        return definition.eval_epsilon
    return fallback


@dataclass
class EpisodeRuntime:
    """
    Mutable counters accumulated during a single episode. Passed by reference to runner
    helpers so they can update state without needing return values on every call.
    """

    optimalPath: list[tuple[int, int]]
    optimalLength: int
    stepLimit: int
    progressPatience: int
    steps: int = 0
    timesHitWall: int = 0
    noProgressSteps: int = 0
    outcome: str = "aborted"
    bestDistance: int = 0
    decisions: int = 0
    optionSelections: int = 0
    optionSteps: int = 0


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

    def runEpisode(self, *, mode: str = "training") -> EpisodeResult:
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

        if hasattr(bot, "reset"):
            bot.reset()

        optimalPath = tools.getOptimalPathInfo(maze.start, maze.end, output="path")
        optimalLength = len(optimalPath)
        stepLimit = compute_step_limit(bot.config, maze, optimalLength)
        progressPatience = compute_no_progress_patience(bot.config, optimalLength, stepLimit)

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
            bot._visProgressPatience = int(progressPatience)
            bot._visSteps = 0
            bot._visTimesHitWall = 0
        return EpisodeRuntime(
            optimalPath=optimalPath,
            optimalLength=optimalLength,
            stepLimit=stepLimit,
            progressPatience=progressPatience,
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
        if getattr(enemyTick, "agent_caught", False):
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
        runtime.decisions += 1
        start_state = bot.state
        semanticAction = decision.actionSpace.semanticId(choice.local_id)
        is_option = bot.isOptionAction(semanticAction)

        if is_option:
            result = bot.applyStep(choice.local_id)
            runtime.optionSelections += 1
            runtime.optionSteps += int(result.duration)
            enemyTick = self._tickEnemies()
            rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
            reward = float(result.reward)
            learningReward = float(result.trainingReward) if result.trainingReward is not None else reward
            if hasattr(rsys, "evaluateSemanticEvents"):
                eventReward = float(rsys.evaluateSemanticEvents(rewardEvents))
                reward += eventReward
                learningReward += eventReward
            runtime.steps += result.duration
            bot.currentEpisodeSteps = int(runtime.steps)
            if mode == "visualization":
                bot._visSteps = int(runtime.steps)
            hitWall = bool(result.info.get("hit_wall", False))
            wasRevisit = bool(result.info.get("was_revisit", False))
            if hitWall:
                runtime.timesHitWall += 1
            else:
                runtime.noProgressSteps = runtime.noProgressSteps + 1 if wasRevisit else 0
            if bot.position != maze.end and runtime.steps >= runtime.stepLimit:
                penalty = float(getattr(bot.config, "stepLimitPenalty", 0.0))
                reward += penalty
                learningReward += penalty
                runtime.outcome = "step_limit"
            if runtime.outcome == "aborted" and runtime.noProgressSteps >= runtime.progressPatience:
                penalty = float(getattr(bot.config, "noProgressPenalty", 0.0))
                reward += penalty
                learningReward += penalty
                runtime.outcome = "no_progress"
            reward = clipRewardValue(float(reward), bot.config)
            learningReward = clipRewardValue(float(learningReward), bot.config)
            bot.totalReward += reward
            if apply_learning:
                bot.applyLearningUpdate(start_state, choice.local_id, learningReward, int(result.duration))
            if runtime.outcome != "aborted":
                return False
            if getattr(enemyTick, "agent_caught", False):
                runtime.outcome = "death_by_enemy"
                return False
            if result.done or bot.position == maze.end:
                runtime.outcome = "goal_reached" if bot.position == maze.end else runtime.outcome
                return False
            return True

        # Attack primitives (semantic IDs >= NUM_PRIMITIVE_DIRECTIONS) don't move the bot.
        # Execute them directly without the movement-validity path.
        if semanticAction >= NUM_PRIMITIVE_DIRECTIONS:
            result = bot.executePrimitiveAction(semanticAction)
            if hasattr(bot, "calculateState"):
                bot.state = bot.calculateState()
            enemyTick = self._tickEnemies()
            rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
            reward = float(result.reward)
            if hasattr(rsys, "evaluateSemanticEvents"):
                reward += float(rsys.evaluateSemanticEvents(rewardEvents))
            if getattr(enemyTick, "agent_caught", False):
                reward = clipRewardValue(reward, bot.config)
                bot.totalReward += reward
                if apply_learning:
                    bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)
                runtime.outcome = "death_by_enemy"
                return False
            runtime.steps += 1
            bot.currentEpisodeSteps = int(runtime.steps)
            if mode == "visualization":
                bot._visSteps = int(runtime.steps)
            if runtime.steps >= runtime.stepLimit:
                reward += float(getattr(bot.config, "stepLimitPenalty", 0.0))
                runtime.outcome = "step_limit"
            reward = clipRewardValue(reward, bot.config)
            bot.totalReward += reward
            if apply_learning:
                bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)
            if runtime.outcome != "aborted":
                return False
            return True

        attempted = bot.tools.calculateNextPosition(bot.position, semanticAction)
        wasReversal = getattr(bot, "previousPosition", None) is not None and attempted == bot.previousPosition
        wasRevisit = attempted in stats.getVisitedPositions()
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
            runtime.timesHitWall += 1
            runtime.steps += 1
            if mode == "visualization":
                bot._visTimesHitWall = int(runtime.timesHitWall)
                bot._visSteps = int(runtime.steps)
            bot.currentEpisodeSteps = int(runtime.steps)
            if runtime.steps >= runtime.stepLimit:
                reward += float(getattr(bot.config, "stepLimitPenalty", 0.0))
                runtime.outcome = "step_limit"
            reward = clipRewardValue(float(reward), bot.config)
            if apply_learning:
                bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)
            bot.totalReward += reward
            if runtime.outcome != "aborted":
                return False
            return True

        result = bot.executePrimitiveAction(semanticAction)
        if hasattr(bot, "calculateState"):
            bot.state = bot.calculateState()
        resultInfo = dict(getattr(result, "info", {}) or {})
        if resultInfo.get("death_by_enemy"):
            reward = clipRewardValue(float(reward + float(result.reward)), bot.config)
            bot.totalReward += reward
            if apply_learning:
                bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)
            runtime.outcome = "death_by_enemy"
            return False
        reward += result.reward
        shapingPenalty = getattr(bot, "shapingPenalty", None)
        if callable(shapingPenalty):
            reward += float(shapingPenalty(wasRevisit, wasReversal))
        enemyTick = self._tickEnemies()
        rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
        if hasattr(rsys, "evaluateSemanticEvents"):
            reward += float(rsys.evaluateSemanticEvents(rewardEvents))

        stats.totalSteps = stats.timesRevisitedSquares + stats.nonRepeatingStepsTaken

        runtime.steps += 1
        bot.currentEpisodeSteps = int(runtime.steps)
        if mode == "visualization":
            bot._visSteps = int(runtime.steps)
        runtime.noProgressSteps = runtime.noProgressSteps + 1 if wasRevisit else 0

        currentDistance = self._manhattan(bot.position, maze.end)
        if currentDistance < runtime.bestDistance:
            runtime.bestDistance = currentDistance
        if getattr(enemyTick, "agent_caught", False):
            runtime.outcome = "death_by_enemy"
        elif bot.position != maze.end and runtime.steps >= runtime.stepLimit:
            reward += float(getattr(bot.config, "stepLimitPenalty", 0.0))
            runtime.outcome = "step_limit"
        elif runtime.noProgressSteps >= runtime.progressPatience:
            reward += float(getattr(bot.config, "noProgressPenalty", 0.0))
            runtime.outcome = "no_progress"
        reward = clipRewardValue(float(reward), bot.config)
        bot.totalReward += reward
        if apply_learning:
            bot.applyLearningUpdate(start_state, choice.local_id, reward, 1)
        if runtime.outcome != "aborted":
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

    def runEvaluationEpisode(self) -> EpisodeResult:
        """Greedy evaluation episode: no Q-table updates, epsilon forced to 0.0."""
        bot = self.bot
        qLearner = getattr(bot, "qLearning", None)
        saved_min_eps = saved_init_eps = saved_steps = None
        if qLearner is not None:
            saved_min_eps = qLearner.minExplorationRate
            saved_init_eps = qLearner.initialExplorationRate
            saved_steps = qLearner.totalSteps
            qLearner.minExplorationRate = 0.0
            qLearner.initialExplorationRate = 0.0
        bot.onEpisodeStart("evaluation")
        runtime = self._startEpisode(mode="evaluation")
        try:
            while bot.position != bot.maze.end:
                if not self._runStep(runtime, mode="evaluation", apply_learning=False):
                    break
            if bot.position == bot.maze.end:
                runtime.outcome = "goal_reached"
            return self._finalizeEvaluationEpisode(runtime)
        finally:
            bot.onEpisodeEnd("evaluation", runtime.outcome)
            if qLearner is not None and saved_min_eps is not None:
                qLearner.minExplorationRate = saved_min_eps
                qLearner.initialExplorationRate = saved_init_eps
                qLearner.totalSteps = saved_steps

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
        persistence = _resolve_episode_persistence_policy(
            bot,
            "training",
            fallback=EpisodePersistencePolicy(
                save_maze_episode=True,
                save_heatmap_stats=True,
                append_reward=True,
            ),
        )
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
            decisions=int(runtime.decisions),
            option_selections=int(runtime.optionSelections),
            option_steps=int(runtime.optionSteps),
            save_maze_episode=bool(persistence.save_maze_episode),
            save_heatmap_stats=bool(persistence.save_heatmap_stats),
            append_reward=bool(persistence.append_reward),
        )

    def _finalizeEvaluationEpisode(self, runtime: EpisodeRuntime) -> EpisodeResult:
        bot = self.bot
        bot.lastEpisodeSuccess = runtime.outcome == "goal_reached"
        bot.lastEpisodeSteps = int(runtime.steps)
        bot.lastEpisodeOptimalSteps = int(runtime.optimalLength)
        bot.lastEpisodeOutcome = str(runtime.outcome)
        bot.lastEpisodeWallHits = int(runtime.timesHitWall)
        bot.lastEpisodeEnemyKills = self._episodeKills
        bot._currentOptimalPath = []
        bot._currentOptimalLength = 0
        heatmapData = dict(bot.statistics.getVisitedPositions())
        persistence = _resolve_episode_persistence_policy(
            bot,
            "evaluation",
            fallback=EpisodePersistencePolicy(
                save_maze_episode=False,
                save_heatmap_stats=False,
                append_reward=False,
            ),
        )
        evalEpsilon = _resolve_evaluation_epsilon(bot, fallback=0.0)
        return EpisodeResult(
            profile_name=str(bot.profileName),
            mode="evaluation",
            outcome=runtime.outcome,
            success=runtime.outcome == "goal_reached",
            total_reward=float(bot.totalReward),
            steps=int(runtime.steps),
            optimal_steps=int(runtime.optimalLength),
            times_hit_wall=int(runtime.timesHitWall),
            heatmap_data=heatmapData,
            maze=bot.maze,
            decisions=int(runtime.decisions),
            option_selections=int(runtime.optionSelections),
            option_steps=int(runtime.optionSteps),
            save_maze_episode=bool(persistence.save_maze_episode),
            save_heatmap_stats=bool(persistence.save_heatmap_stats),
            append_reward=bool(persistence.append_reward),
            eval_epsilon=evalEpsilon,
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
    decisions: int = 0
    optionSelections: int = 0
    optionSteps: int = 0


class DQNEpisodeRunner:
    def __init__(self, bot: Any):
        self.bot = bot
        self._enemySystem = EnemySystem()
        self._pendingKills: int = 0
        self._episodeKills: int = 0
        context = getattr(bot, "_context", None)
        if context is not None:
            context.on(ENEMY_KILLED, lambda **_: self._onEnemyKilled())

    def runEpisode(self, *, mode: str = "training") -> EpisodeResult:
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

    def runEvaluationEpisode(self) -> EpisodeResult:
        return self.runEpisode(mode="evaluation")

    def _startEpisode(self) -> DQNEpisodeRuntime:
        bot = self.bot
        optimalPathRaw = bot.tools.getOptimalPathInfo(bot.maze.start, bot.maze.end, output="path")
        optimalPath: list[tuple[int, int]] = cast(list[tuple[int, int]], optimalPathRaw) if isinstance(optimalPathRaw, list) else []
        optimalLength = len(optimalPath)
        cfg = bot.config
        stepLimit = compute_step_limit(cfg, bot.maze, optimalLength)
        progressPatience = compute_no_progress_patience(cfg, optimalLength, stepLimit)
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
        if getattr(enemyTick, "agent_caught", False):
            events.append(RewardEvent(name="death_by_enemy", payload={}))
        return tuple(events)

    def _runStep(self, runtime: DQNEpisodeRuntime, *, mode: str) -> bool:
        if getattr(self.bot, "usesHierarchicalPolicy", False):
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
                forceEpsilon=0.0,
                allowWarmupPlanner=False,
            )
        )
        runtime.decisions += 1

        result = bot.applyStep(choice.local_id, training=(mode == "training"))
        reward = float(result.reward)
        learningReward = float(result.trainingReward) if result.trainingReward is not None else reward
        if result.info.get("death_by_enemy"):
            runtime.outcome = "death_by_enemy"
        enemyTick = self._tickEnemies()
        rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
        if hasattr(bot.rewardSystem, "evaluateSemanticEvents"):
            eventReward = float(bot.rewardSystem.evaluateSemanticEvents(rewardEvents))
            reward += eventReward
            learningReward += eventReward
        hitWall = bool(result.info.get("hit_wall", False))
        wasRevisit = bool(result.info.get("was_revisit", False))
        if result.info.get("option"):
            runtime.optionSelections += 1
            runtime.optionSteps += int(result.duration)
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
            penalty = float(bot.config.stepLimitPenalty)
            reward += penalty
            learningReward += penalty
            runtime.outcome = "step_limit"
            done = True
        if not done and getattr(enemyTick, "agent_caught", False):
            runtime.outcome = "death_by_enemy"
            done = True
        if not done and runtime.noProgressSteps >= runtime.progressPatience:
            penalty = float(bot.config.noProgressPenalty)
            reward += penalty
            learningReward += penalty
            runtime.outcome = "no_progress"
            done = True
        reward = clipRewardValue(float(reward), bot.config)
        learningReward = clipRewardValue(float(learningReward), bot.config)

        if mode == "training":
            bot.applyLearningUpdate(encodedState, choice.local_id, learningReward, done, int(result.duration))
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
                forceEpsilon=0.0,
                allowWarmupPlanner=False,
            )
        )
        runtime.decisions += 1
        result = bot.executeHierarchicalOption(choice.local_id, training=(mode == "training"))
        reward = float(result.reward)
        learningReward = float(result.trainingReward) if result.trainingReward is not None else reward
        if result.info.get("death_by_enemy"):
            runtime.outcome = "death_by_enemy"
        enemyTick = self._tickEnemies()
        rewardEvents = self._rewardEventsFromEnemyTick(enemyTick) + self._collectKillEvents()
        if hasattr(bot.rewardSystem, "evaluateSemanticEvents"):
            eventReward = float(bot.rewardSystem.evaluateSemanticEvents(rewardEvents))
            reward += eventReward
            learningReward += eventReward
        hitWall = bool(result.info.get("hit_wall", False))
        wasRevisit = bool(result.info.get("was_revisit", False))
        if result.info.get("option"):
            runtime.optionSelections += 1
            runtime.optionSteps += int(result.duration)
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
            penalty = float(bot.config.stepLimitPenalty)
            reward += penalty
            learningReward += penalty
            runtime.outcome = "step_limit"
            done = True
        if not done and getattr(enemyTick, "agent_caught", False):
            runtime.outcome = "death_by_enemy"
            done = True
        if not done and runtime.noProgressSteps >= runtime.progressPatience:
            penalty = float(bot.config.noProgressPenalty)
            reward += penalty
            learningReward += penalty
            runtime.outcome = "no_progress"
            done = True
        reward = clipRewardValue(float(reward), bot.config)
        learningReward = clipRewardValue(float(learningReward), bot.config)

        if mode == "training":
            bot.applyHighLevelLearningUpdate(encodedState, choice.local_id, learningReward, done, int(result.duration))
        bot.addReward(reward)
        return not done

    def _finalizeEpisode(self, runtime: DQNEpisodeRuntime, *, mode: str) -> EpisodeResult:
        bot = self.bot
        success = runtime.outcome == "goal_reached"
        is_warming_up = mode == "training" and bool(bot.isWarmingUp)
        evalEpsilon = _resolve_evaluation_epsilon(bot, fallback=0.0 if mode == "evaluation" else None)

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
                success=success,
                total_reward=float(bot.totalReward),
                steps=int(runtime.steps),
                optimal_steps=int(runtime.optimalLength),
                times_hit_wall=int(runtime.timesHitWall),
                heatmap_data=heatmapData,
                maze=bot.maze,
                decisions=int(runtime.decisions),
                option_selections=int(runtime.optionSelections),
                option_steps=int(runtime.optionSteps),
                save_maze_episode=False,
                save_heatmap_stats=False,
                append_reward=False,
                is_warming_up=True,
                eval_epsilon=evalEpsilon,
            )

        persistence = _resolve_episode_persistence_policy(
            bot,
            mode,
            fallback=EpisodePersistencePolicy(
                save_maze_episode=(mode == "evaluation"),
                save_heatmap_stats=(mode == "training"),
                append_reward=(mode == "evaluation"),
            ),
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
            decisions=int(runtime.decisions),
            option_selections=int(runtime.optionSelections),
            option_steps=int(runtime.optionSteps),
            save_maze_episode=bool(persistence.save_maze_episode),
            save_heatmap_stats=bool(persistence.save_heatmap_stats),
            append_reward=bool(persistence.append_reward),
            is_warming_up=False,
            eval_epsilon=evalEpsilon,
        )
