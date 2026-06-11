from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EpisodePersistencePolicy:
    """Controls which durable artifacts should be persisted for an episode."""

    save_maze_episode: bool = False
    save_heatmap_stats: bool = False
    append_reward: bool = False


@dataclass(frozen=True)
class EvaluationEpisodeDefinition:
    """Bot-owned definition of what its evaluation episode means."""

    description: str = ""
    frequency: int = 0
    is_dedicated_episode: bool = False
    eval_epsilon: float | None = None
    persistence: EpisodePersistencePolicy = field(default_factory=EpisodePersistencePolicy)


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
    decisions: int = 0
    option_selections: int = 0
    option_steps: int = 0

    # Persistence control resolved from the bot's episode policy and copied
    # onto the immutable result so the recorder stays generic.
    save_maze_episode: bool = True
    save_heatmap_stats: bool = True
    append_reward: bool = True
    is_warming_up: bool = False       # DQN warmup: recorder skips all persistence
    eval_epsilon: float | None = None
