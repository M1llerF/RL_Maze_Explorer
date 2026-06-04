from __future__ import annotations

from typing import Any

from services.episode_result import EpisodeResult
from services.repository import ArtifactsRepository


class PostEpisodeRecorder:
    """Handles all post-episode artifact persistence.

    Runners return an EpisodeResult. Bots call recorder.record(self, result)
    so that repo writes, reward logs, and bot-specific artifact saves
    (Q-table, checkpoint) never appear inside runner code.
    """

    def __init__(self, repository: ArtifactsRepository) -> None:
        self.repository = repository

    def record(self, bot: Any, result: EpisodeResult) -> None:
        """Persist all artifacts described by result. No-op during warmup."""
        if result.is_warming_up:
            return

        repo = self.repository
        profile = result.profile_name

        try:
            if result.save_maze_episode:
                repo.saveMazeEpisode(profile, result.maze, result.heatmap_data, result.total_reward)
            if result.save_heatmap_stats:
                repo.updateStepsFromHeatmap(profile, result.heatmap_data)
                if result.times_hit_wall:
                    repo.incrementTimesHitWall(profile, result.times_hit_wall)
        except Exception as e:
            print(f"[EpisodeFinalize] Artifact persistence failed for '{profile}': {e}")

        if result.append_reward:
            try:
                repo.appendReward(profile, result.total_reward)
            except Exception as e:
                print(f"[EpisodeFinalize] Reward log failed for '{profile}': {e}")

        if result.mode == "training":
            bot.persistTrainingArtifacts()
