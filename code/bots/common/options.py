from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.context import EnvironmentContext


class OptionSpec:
    """
    Abstract base for reusable macro-actions / options.

    Subclass this and override canStart, shouldTerminate, choosePrimitiveAction,
    and optionally intrinsicReward.

    Usage:
        registry.registerOption(MoveRightUntilJunction())
    """

    def __init__(
        self,
        id: int,
        name: str,
        max_duration: int = 20,
        description: str = "",
    ) -> None:
        self.id = id
        self.name = name
        self.kind = "option"
        self.description = description
        self.max_duration = max_duration

    def canStart(self, context: EnvironmentContext) -> bool:
        return True

    def shouldTerminate(self, context: EnvironmentContext, steps_elapsed: int) -> bool:
        return steps_elapsed >= self.max_duration

    def choosePrimitiveAction(self, context: EnvironmentContext) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} must implement choosePrimitiveAction")

    def intrinsicReward(self, context: EnvironmentContext, transition: Any) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, name={self.name!r})"
