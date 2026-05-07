from __future__ import annotations

from typing import Any


class BaseBot:
    """
    Base lifecycle contract for training and visualization.

    Training lifecycle:
    - `run_episode()` is a full training episode entrypoint.
    - Runners may call `on_episode_start(...)`, `on_episode_step(...)`,
      and `on_episode_end(...)` hooks around episode execution.
    - Training updates and persistence are allowed only in training flows.

    Visualization lifecycle:
    - Visualization must remain inference-only and non-persistent.
    - Bots can expose step-wise visualization methods, but these must not
      mutate durable training artifacts.
    """
    def __init__(self, maze: Any, statistics: Any, config: Any = None) -> None:
        """
        Initialize the base bot.

        :param maze: The maze object that the bot will navigate.
        :param statistics: a instance of BotStatistics to track the bot's performance.
        :param config: Optional configuration for specific algorithms (e.g., Q-learning config).
        """
        self.maze = maze
        self.statistics = statistics
        self.config = config
        # Cooperative stop flag to abort long episodes promptly
        self._stop_requested = False
    
    def reset(self) -> None:
        """Reset the bot's state and statistics. Should be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def calculate_state(self) -> Any:
        """Calculate the current state of the bot. Should be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def run_episode(self) -> None:
        """Run a single episode of the bot's operation. Should be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    # Episode lifecycle hooks ------------------------------------------------
    def on_episode_start(self, mode: str) -> None:
        """
        Hook called exactly once at episode start.

        :param mode: Execution context such as "training" or "visualization".
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def on_episode_step(self, mode: str, step_index: int) -> None:
        """
        Hook called on each episode loop iteration.

        :param mode: Execution context such as "training" or "visualization".
        :param step_index: Zero-based step index.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def on_episode_end(self, mode: str, outcome: str) -> None:
        """
        Hook called exactly once when an episode exits.

        :param mode: Execution context such as "training" or "visualization".
        :param outcome: Human-readable terminal reason.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    # Cooperative stop handling
    def request_stop(self) -> None:
        self._stop_requested = True

    def clear_stop(self) -> None:
        self._stop_requested = False
