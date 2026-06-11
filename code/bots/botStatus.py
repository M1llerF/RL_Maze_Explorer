from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BotStatus:
    """Normalized snapshot of a bot's runtime state for UI consumption.

    All bot types map their internals into this stable shape.
    UI reads only this object — never bot.agent.*, bot.qLearning.*, etc.
    """

    episode_count: int
    is_warming_up: bool
    replay_size: int              # current replay buffer entries; 0 for Q-learning
    warmup_required: int          # replayWarmupSteps for DQN; 0 for Q-learning
    current_exploration_rate: float  # epsilon (DQN) or exploration rate (Q-learning)
    epsilon_is_manual: bool       # True when a manual epsilon override is active (DQN only)
    checkpoint_frequency: int     # 0 = autosave disabled; DQN only
    current_episode_steps: int
    last_episode_success: bool
    artifact_save_label: str      # "Q-table" (QL) or "checkpoint" (DQN) — for save status UI
