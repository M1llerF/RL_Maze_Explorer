# bot_configs.py
from __future__ import annotations

import inspect
from typing import Any

class QLearningConfig:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        use_position_in_state: bool = True,
    ) -> None:
        """
        Initialize Q-learning configuration with default learning rate and discount factor.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # If False, the Q-state key excludes absolute position, aiding generalization across mazes
        self.use_position_in_state = use_position_in_state

    def customize(self) -> None:
        """
        Customize Q-learning parameters via user input.
        Prompts the user to enter new values for learning rate and discount factor.
        If the user input is invalid or left blank, the default values are used.
        """
        self.learning_rate = self._get_float_input("Enter learning rate (default 0.1): ", self.learning_rate)
        self.discount_factor = self._get_float_input("Enter discount factor (default 0.9): ", self.discount_factor)

    @staticmethod
    def _get_float_input(prompt: str, default: float) -> float:
        """
        Helper method to get a float input from the user.
        If the input is invalid or left blank, the default value is returned.

        :param prompt: The prompt message for input.
        :param default: The default value to return if input is invalid or blank.
        :return: The user input as a float or the default value.
        """
        try:
            return float(input(prompt) or default)
        except ValueError:
            print(f"Invalid input. Using default value {default}.")
            return default

# Define configurations for different bot types
bot_configs: dict[str, dict[str, Any]] = {
    "QLearningBot": {
        "class": QLearningConfig,
        "params": {
            "Learning Rate": "learning_rate",
            "Discount Factor": "discount_factor"
        },
        "rewards": {
            'goal_reached': 1000,
            'hit_wall': -100,
            'revisit_optimal_path': -10,
            'revisit_non_optimal_path': -15,
            'move_in_optimal_path': 5,
            'see_goal_new_location': 50,
            'see_goal_revisit': 5,
            'per_move_penalty': -1
        }
    },
}


def get_config_class_for_bot_type(bot_type: str | None) -> type[Any]:
    """Return the config class registered for a bot type, defaulting to QLearningConfig."""
    if bot_type and bot_type in bot_configs:
        config_cls = bot_configs[bot_type].get("class")
        if isinstance(config_cls, type):
            return config_cls
    return QLearningConfig


def build_config_for_bot_type(bot_type: str | None, raw_config: Any) -> Any:
    """Build a bot config instance from raw profile payload data."""
    config_cls = get_config_class_for_bot_type(bot_type)
    if isinstance(raw_config, config_cls):
        return raw_config

    cfg_dict: dict[str, Any] = raw_config if isinstance(raw_config, dict) else {}
    try:
        signature = inspect.signature(config_cls)
        allowed = {
            name
            for name in signature.parameters
            if name != "self"
            and signature.parameters[name].kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
    except (TypeError, ValueError):
        allowed = set()

    kwargs = {str(k): v for k, v in cfg_dict.items() if str(k) in allowed}
    try:
        return config_cls(**kwargs)
    except TypeError:
        return config_cls()
