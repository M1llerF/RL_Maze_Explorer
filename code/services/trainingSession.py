from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from services.curriculum import CurriculumConfig, CurriculumDecision, CurriculumManager


@dataclass
class TrainingSession:
    """Owns mutable state for one training or warmup run."""

    selected_profile: str
    maze_mode: str = "Random"
    pool_size: int = 20
    random_min_generation_length: int | None = None
    random_max_generation_length: int | None = None
    random_include_enemies: bool = False
    total_rounds: int | None = None
    is_collecting: bool = False
    is_training: bool = False
    bot_index: int | None = None
    completed_rounds: int = 0
    curriculum_enabled: bool = False
    curriculum: CurriculumManager | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event)

    def configure_environment(self, env: Any) -> None:
        env.configureMazeMode(
            self.maze_mode,
            poolSize=int(self.pool_size),
            minLength=self.random_min_generation_length,
            maxLength=self.random_max_generation_length,
            includeEnemies=self.random_include_enemies,
        )
        self.curriculum_enabled = getattr(env, "curriculumActive", False)
        self.curriculum = None
        if self.curriculum_enabled:
            max_maze_size = (
                int(self.random_max_generation_length)
                if self.random_max_generation_length is not None
                else 31
            )
            self.curriculum = CurriculumManager(
                CurriculumConfig(
                    windowSize=20,
                    minSuccessRateToGraduate=0.7,
                    maxAStarRatioToGraduate=2.0,
                    maxMazeSize=max_maze_size,
                )
            )
            width, height = env.getMazeSize()
            start_size = min(int(width), int(height))
            self.curriculum.resetProfile(self.selected_profile, initialSize=start_size)

    def bind_bot_index(self, bot_index: int) -> None:
        self.bot_index = int(bot_index)

    def clear_bot_index(self) -> None:
        self.bot_index = None

    def request_stop(self) -> None:
        self._stop_event.set()

    def clear_stop(self) -> None:
        self._stop_event.clear()

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def mark_episode_completed(self) -> None:
        self.completed_rounds += 1

    def apply_curriculum_update(self, env: Any, bot: Any) -> CurriculumDecision | None:
        if not self.curriculum_enabled or self.curriculum is None:
            return None

        success = getattr(bot, "lastEpisodeSuccess", False)
        episode_steps = int(getattr(bot, "lastEpisodeSteps", 0))
        astar_steps = int(getattr(bot, "lastEpisodeOptimalSteps", 0))
        decision = self.curriculum.update(
            self.selected_profile,
            success=success,
            episodeSteps=episode_steps,
            astarSteps=astar_steps,
        )
        if decision.graduated:
            next_size = int(decision.nextSize)
            env.setMazeSize(next_size, next_size)
        return decision
