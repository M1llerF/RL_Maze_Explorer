from __future__ import annotations

import time
from typing import Any, Callable


class QLearningWarmupCollector:
    """
    Runs a Q-learning bot in planner-guided mode to pre-warm its Q-table.

    Unlike the DQN WarmupCollector there is no separate store to inject —
    the Q-table itself accumulates knowledge as guided episodes run.  The
    caller is responsible for saving the Q-table afterward via
    bot.persistTrainingArtifacts().
    """

    def collect(
        self,
        bot: Any,
        env: Any,
        botIndex: int,
        nEpisodes: int = 20,
        onProgress: Callable[[int, int], None] | None = None,
        onEpisodeComplete: Callable[[Any], None] | None = None,
        stopRequested: Callable[[], bool] | None = None,
    ) -> None:
        """
        Run nEpisodes of planner-guided exploration, updating the Q-table.

        onProgress receives (completed, total) after each episode.
        stopRequested, if provided, is checked before each episode.
        """
        target = max(1, int(nEpisodes))
        previousWarmupMode = getattr(bot, "_collectingWarmup", False)
        bot._collectingWarmup = True

        completed = 0
        try:
            while completed < target:
                if stopRequested is not None and stopRequested():
                    break
                bot.runEpisode()
                completed += 1
                if onEpisodeComplete is not None:
                    onEpisodeComplete(bot)
                if onProgress is not None:
                    onProgress(completed, target)
                env.resetEnvironment(botIndex)
                time.sleep(0.01)
        finally:
            bot._collectingWarmup = previousWarmupMode
