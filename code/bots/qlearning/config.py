from __future__ import annotations

from typing import Any


class QLearningConfig:
    def __init__(
        self,
        learningRate: float = 0.1,
        discountFactor: float = 0.9,
        usePositionInState: bool = True,
    ) -> None:
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.usePositionInState = usePositionInState

    def customize(self) -> None:
        self.learningRate = self._getFloatInput("Enter learning rate (default 0.1): ", self.learningRate)
        self.discountFactor = self._getFloatInput("Enter discount factor (default 0.9): ", self.discountFactor)

    @staticmethod
    def _getFloatInput(prompt: str, default: float) -> float:
        try:
            return float(input(prompt) or default)
        except ValueError:
            print(f"Invalid input. Using default value {default}.")
            return default


BOT_SPEC: dict[str, Any] = {
    "type": "QLearningBot",
    "class": QLearningConfig,
    "params": {
        "Learning Rate": "learningRate",
        "Discount Factor": "discountFactor",
    },
    "rewards": {
        "goal_reached": 1000,
        "hit_wall": -100,
        "revisit_optimal_path": -10,
        "revisit_non_optimal_path": -15,
        "move_in_optimal_path": 5,
        "see_goal_new_location": 50,
        "see_goal_revisit": 5,
        "per_move_penalty": -1,
    },
}
