# bot_configs.py
from __future__ import annotations

import inspect
from typing import Any

class QLearningConfig:
    def __init__(
        self,
        learningRate: float = 0.1,
        discountFactor: float = 0.9,
        usePositionInState: bool = True,
    ) -> None:
        """
        Initialize Q-learning configuration with default learning rate and discount factor.
        """
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        # If False, the Q-state key excludes absolute position, aiding generalization across mazes
        self.usePositionInState = usePositionInState

    def customize(self) -> None:
        """
        Customize Q-learning parameters via user input.
        Prompts the user to enter new values for learning rate and discount factor.
        If the user input is invalid or left blank, the default values are used.
        """
        self.learningRate = self._getFloatInput("Enter learning rate (default 0.1): ", self.learningRate)
        self.discountFactor = self._getFloatInput("Enter discount factor (default 0.9): ", self.discountFactor)

    @staticmethod
    def _getFloatInput(prompt: str, default: float) -> float:
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
botConfigs: dict[str, dict[str, Any]] = {
    "QLearningBot": {
        "class": QLearningConfig,
        "params": {
            "Learning Rate": "learningRate",
            "Discount Factor": "discountFactor"
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


def getConfigClassForBotType(botType: str | None) -> type[Any]:
    """Return the config class registered for a bot type, defaulting to QLearningConfig."""
    if botType and botType in botConfigs:
        configCls = botConfigs[botType].get("class")
        if isinstance(configCls, type):
            return configCls
    return QLearningConfig


def buildConfigForBotType(botType: str | None, rawConfig: Any) -> Any:
    """Build a bot config instance from raw profile payload data."""
    configCls = getConfigClassForBotType(botType)
    if isinstance(rawConfig, configCls):
        return rawConfig

    cfgDict: dict[str, Any] = rawConfig if isinstance(rawConfig, dict) else {}
    try:
        signature = inspect.signature(configCls)
        allowed = {
            name
            for name in signature.parameters
            if name != "self"
            and signature.parameters[name].kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
    except (TypeError, ValueError):
        allowed = set()

    normalized: dict[str, Any] = {}
    for k, v in cfgDict.items():
        key = str(k)
        normalized[key] = v
        # Backward compatibility for legacy snake_case profile payload keys.
        if "_" in key:
            parts = [p for p in key.split("_") if p]
            if parts:
                camel = parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])
                normalized.setdefault(camel, v)

    kwargs = {k: v for k, v in normalized.items() if k in allowed}
    try:
        return configCls(**kwargs)
    except TypeError:
        return configCls()
