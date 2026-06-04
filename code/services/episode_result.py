from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeResult:
    """Immutable record of a completed episode returned by a runner.

    Runners populate this and return it. PostEpisodeRecorder consumes it
    to handle all artifact persistence, keeping runners free of repo concerns.
    """

    profile_name: str
    mode: str               # "training" or "evaluation"
    outcome: str            # "goal_reached", "step_limit", "no_progress", etc.
    success: bool
    total_reward: float
    steps: int
    optimal_steps: int
    times_hit_wall: int
    heatmap_data: dict[tuple[int, int], int]
    maze: Any               # maze object forwarded to saveMazeEpisode

    # Persistence control — set by the runner based on bot type and mode.
    # Allows the recorder to be generic without knowing bot-type specifics.
    save_maze_episode: bool = True    # True for QL training and DQN evaluation
    save_heatmap_stats: bool = True   # True for QL training and DQN training
    append_reward: bool = True        # True for QL training and DQN evaluation
    is_warming_up: bool = False       # DQN warmup: recorder skips all persistence
