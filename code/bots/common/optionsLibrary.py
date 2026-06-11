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
_MOMENTUM_BASE = _OPTION_BASE + 10   # 110-119
_ASTAR_OPTION_ID = _OPTION_BASE + 20  # 120
_DELTA_TO_DIRECTION = {v: k for k, v in DIRECTION_DELTAS.items() if k in (UP, DOWN, LEFT, RIGHT)}
_ALL_DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


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


class AStarShortestPathToDecision(OptionSpec):
    """A* (full map knowledge) upper-bound option: follow shortest path to next decision point."""

    def __init__(self) -> None:
        super().__init__(
            id=_ASTAR_OPTION_ID,
            name="astar_shortest_path_to_next_decision",
            max_duration=40,
            description="A* macro: follow full-maze shortest path until next junction, dead-end, or goal",
        )

    def canStart(self, context: EnvironmentContext) -> bool:
        return not context.isAtGoal() and self._nextDirection(context) is not None

    def shouldTerminate(self, context: EnvironmentContext, steps_elapsed: int) -> bool:
        return (
            steps_elapsed >= self.max_duration
            or context.isAtGoal()
            or (steps_elapsed > 0 and (context.isAtJunction() or context.isAtDeadEnd()))
            or self._nextDirection(context) is None
        )

    def choosePrimitiveAction(self, context: EnvironmentContext) -> int:
        direction = self._nextDirection(context)
        if direction is None:
            return UP
        return direction

    def _nextDirection(self, context: EnvironmentContext) -> int | None:
        from pathfinding import Pathfinding

        path = Pathfinding.bfsShortestPathGrid(
            context.maze.grid,
            context.position,
            context.goalPosition,
        )
        if len(path) < 2:
            return None
        next_pos = path[1]
        delta = (next_pos[0] - context.position[0], next_pos[1] - context.position[1])
        direction = _DELTA_TO_DIRECTION.get(delta)
        if direction is None or not context.canMove(direction):
            return None
        return direction


class MomentumCorridorMacro(OptionSpec):
    """
    Oracle-free directional scouting macro.

    Begins moving in a fixed direction, follows corridor momentum, and makes a
    small local-memory choice at junctions by preferring adjacent cells the bot
    has personally visited least often in this episode.  This gives the macro
    more large-maze utility without adding separate navigation policies.

    Uses only local movement-validity information, adjacent target coordinates,
    and per-episode visit counts — no A*, BFS, full maze layout, or global map.
    """

    MAX_MACRO_STEPS: int = 12
    MAX_LOCAL_CHOICES: int = 2

    def __init__(self, direction: int, option_id: int) -> None:
        dir_names = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
        super().__init__(
            id=option_id,
            name=f"momentum_corridor_{dir_names[direction]}",
            max_duration=self.MAX_MACRO_STEPS,
            description=(
                f"Directional scout {dir_names[direction]}: follow corridor momentum "
                "and prefer locally least-visited branches"
            ),
        )
        self._initial_direction = direction
        self._current_direction: int = direction
        self._visited: set[tuple[int, int]] = set()
        self._local_choices: int = 0

    def _reset(self, context: EnvironmentContext) -> None:
        self._current_direction = self._initial_direction
        self._visited = {context.position}
        self._local_choices = 0

    def canStart(self, context: EnvironmentContext) -> bool:
        return _safeCanMove(context, self._initial_direction) and not context.isAtGoal()

    def shouldTerminate(self, context: EnvironmentContext, steps_elapsed: int) -> bool:
        if steps_elapsed == 0:
            self._reset(context)
            return not _safeCanMove(context, self._initial_direction)

        pos = context.position
        if pos in self._visited:
            return True
        self._visited.add(pos)

        if steps_elapsed >= self.max_duration or context.isAtGoal():
            return True

        candidates = self._candidateDirections(context)
        if not candidates:
            return True
        if self._local_choices >= self.MAX_LOCAL_CHOICES and self._isChoicePoint(context, candidates):
            return True

        return False

    def choosePrimitiveAction(self, context: EnvironmentContext) -> int:
        candidates = self._candidateDirections(context)
        if not candidates:
            return self._current_direction

        visits = _visitedCounts(context)
        previous = getattr(context.bot, "previousPosition", None)

        def score(direction: int) -> tuple[int, int, int, int, int]:
            dr, dc = DIRECTION_DELTAS[direction]
            target = (context.position[0] + dr, context.position[1] + dc)
            visit_count = int(visits.get(target, 0))
            backtrack_penalty = 1 if previous is not None and tuple(previous) == target else 0
            turn_penalty = _turnCost(self._current_direction, direction)
            initial_bias = 0 if direction == self._initial_direction else 1
            return (visit_count, backtrack_penalty, turn_penalty, initial_bias, direction)

        if self._isChoicePoint(context, candidates):
            self._local_choices += 1
        self._current_direction = min(candidates, key=score)
        return self._current_direction

    def _candidateDirections(self, context: EnvironmentContext) -> list[int]:
        return [
            d for d in _ALL_DIRECTIONS
            if _safeCanMove(context, d) and _adjacentPosition(context, d) not in self._visited
        ]

    def _isChoicePoint(self, context: EnvironmentContext, candidates: list[int]) -> bool:
        previous = getattr(context.bot, "previousPosition", None)
        non_backtracking = [
            d for d in candidates
            if previous is None or _adjacentPosition(context, d) != tuple(previous)
        ]
        return len(non_backtracking) >= 2


class MomentumCorridorRight(MomentumCorridorMacro):
    def __init__(self) -> None:
        super().__init__(direction=RIGHT, option_id=_MOMENTUM_BASE + 0)


class MomentumCorridorLeft(MomentumCorridorMacro):
    def __init__(self) -> None:
        super().__init__(direction=LEFT, option_id=_MOMENTUM_BASE + 1)


class MomentumCorridorUp(MomentumCorridorMacro):
    def __init__(self) -> None:
        super().__init__(direction=UP, option_id=_MOMENTUM_BASE + 2)


class MomentumCorridorDown(MomentumCorridorMacro):
    def __init__(self) -> None:
        super().__init__(direction=DOWN, option_id=_MOMENTUM_BASE + 3)


def _safeCanMove(context: EnvironmentContext, direction: int) -> bool:
    return bool(context.canMove(direction)) and not _enemyAhead(context, direction)


def _adjacentPosition(context: EnvironmentContext, direction: int) -> tuple[int, int]:
    dr, dc = DIRECTION_DELTAS[direction]
    return context.position[0] + dr, context.position[1] + dc


def _enemyAhead(context: EnvironmentContext, direction: int) -> bool:
    from environment.entities import Enemy

    target = _adjacentPosition(context, direction)
    return any(
        isinstance(e, Enemy) and e._alive
        for e in context.entityRegistry.entitiesAt(target)
    )


def _visitedCounts(context: EnvironmentContext) -> dict[tuple[int, int], int]:
    stats = getattr(context.bot, "statistics", None)
    getter = getattr(stats, "getVisitedPositions", None)
    if callable(getter):
        return dict(getter())
    return {}


def _turnCost(current_direction: int, candidate_direction: int) -> int:
    if candidate_direction == current_direction:
        return 0
    if candidate_direction == _OPPOSITE.get(current_direction):
        return 2
    return 1


def naiveOptions() -> list[OptionSpec]:
    """Return the directional corridor macro set."""
    return [
        MoveRightUntilJunction(),
        MoveLeftUntilJunction(),
        MoveUpUntilJunction(),
        MoveDownUntilJunction(),
    ]


def momentumOptions() -> list[OptionSpec]:
    """Return the four symmetric fair local-memory directional macros."""
    return [
        MomentumCorridorRight(),
        MomentumCorridorLeft(),
        MomentumCorridorUp(),
        MomentumCorridorDown(),
    ]


def defaultOptions(option_set: str = "naive") -> list[OptionSpec]:
    """Return built-in macro options for the requested research option set."""
    if str(option_set) == "momentum":
        return momentumOptions()
    options = naiveOptions()
    if str(option_set) == "astar":
        options.append(AStarShortestPathToDecision())
    return options
