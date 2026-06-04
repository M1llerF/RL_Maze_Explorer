from __future__ import annotations

from typing import TYPE_CHECKING

from .actions import UP, DOWN, LEFT, RIGHT, DIRECTION_DELTAS
from .options import OptionSpec

if TYPE_CHECKING:
    from environment.context import EnvironmentContext

# Option ID ranges:
# Primitive actions occupy IDs 0-3.
# Built-in options use IDs 100-199.
# User-defined options should start at 200+.

_OPTION_BASE = 100


class MoveUntilJunction(OptionSpec):
    """Move in a fixed direction until a junction, dead-end, or goal is reached."""

    def __init__(self, direction: int, option_id: int) -> None:
        dir_names = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
        name = f"move_{dir_names[direction]}_until_junction"
        super().__init__(
            id=option_id,
            name=name,
            max_duration=40,
            description=f"Move {dir_names[direction]} until junction, wall, or goal",
        )
        self._direction = direction

    def canStart(self, context: EnvironmentContext) -> bool:
        return context.canMove(self._direction) and not context.isAtGoal()

    def shouldTerminate(self, context: EnvironmentContext, steps_elapsed: int) -> bool:
        return (
            steps_elapsed >= self.max_duration
            or not context.canMove(self._direction)
            or (steps_elapsed > 0 and context.isAtJunction())
            or context.isAtGoal()
            or self._enemyAhead(context)
        )

    def _enemyAhead(self, context: EnvironmentContext) -> bool:
        """Return True if an alive enemy occupies the next cell in our direction."""
        from environment.entities import Enemy
        dr, dc = DIRECTION_DELTAS[self._direction]
        target = (context.position[0] + dr, context.position[1] + dc)
        return any(
            isinstance(e, Enemy) and e._alive
            for e in context.entityRegistry.entitiesAt(target)
        )

    def choosePrimitiveAction(self, context: EnvironmentContext) -> int:
        return self._direction


class MoveRightUntilJunction(MoveUntilJunction):
    def __init__(self) -> None:
        super().__init__(direction=RIGHT, option_id=_OPTION_BASE + 0)


class MoveLeftUntilJunction(MoveUntilJunction):
    def __init__(self) -> None:
        super().__init__(direction=LEFT, option_id=_OPTION_BASE + 1)


class MoveUpUntilJunction(MoveUntilJunction):
    def __init__(self) -> None:
        super().__init__(direction=UP, option_id=_OPTION_BASE + 2)


class MoveDownUntilJunction(MoveUntilJunction):
    def __init__(self) -> None:
        super().__init__(direction=DOWN, option_id=_OPTION_BASE + 3)


def defaultOptions() -> list[OptionSpec]:
    """Return the built-in directional macro set."""
    return [
        MoveRightUntilJunction(),
        MoveLeftUntilJunction(),
        MoveUpUntilJunction(),
        MoveDownUntilJunction(),
    ]
