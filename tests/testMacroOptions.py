from __future__ import annotations

import pytest

from bots.common.actions import DOWN, LEFT, RIGHT, UP, DIRECTION_DELTAS
from bots.common.optionsLibrary import (
    MomentumCorridorMacro,
    MomentumCorridorRight,
    defaultOptions,
)
from bots.dqnlearning.config import ConfigValidationError, DQNConfig


class _FakeEntityRegistry:
    def entitiesAt(self, _position: tuple[int, int]) -> list[object]:
        return []


class _FakeStats:
    def __init__(self) -> None:
        self.visited: dict[tuple[int, int], int] = {}

    def getVisitedPositions(self) -> dict[tuple[int, int], int]:
        return self.visited


class _FakeBot:
    def __init__(self) -> None:
        self.previousPosition: tuple[int, int] | None = None
        self.statistics = _FakeStats()


class _FakeContext:
    def __init__(
        self,
        grid: dict[tuple[int, int], set[int]],
        goal: tuple[int, int] = (99, 99),
    ) -> None:
        self._grid = grid
        self._goal = goal
        self.position: tuple[int, int] = (0, 0)
        self.bot = _FakeBot()
        self.entityRegistry = _FakeEntityRegistry()

    def canMove(self, direction: int) -> bool:
        return direction in self._grid.get(self.position, set())

    def isAtGoal(self) -> bool:
        return self.position == self._goal

    def isAtJunction(self) -> bool:
        return sum(1 for direction in (UP, DOWN, LEFT, RIGHT) if self.canMove(direction)) >= 3


def _run_macro(macro: MomentumCorridorMacro, ctx: _FakeContext, max_steps: int = 50) -> tuple[list[tuple[int, int]], int]:
    path = [ctx.position]
    steps = 0
    while not macro.shouldTerminate(ctx, steps):
        direction = macro.choosePrimitiveAction(ctx)
        dr, dc = DIRECTION_DELTAS[direction]
        previous = ctx.position
        ctx.position = (ctx.position[0] + dr, ctx.position[1] + dc)
        ctx.bot.previousPosition = previous
        ctx.bot.statistics.visited[ctx.position] = ctx.bot.statistics.visited.get(ctx.position, 0) + 1
        path.append(ctx.position)
        steps += 1
        if steps > max_steps:
            raise AssertionError(f"Macro did not terminate within {max_steps} steps")
    return path, steps


def test_momentum_macro_prefers_least_visited_branch() -> None:
    grid = {
        (0, 0): {RIGHT},
        (0, 1): {DOWN, UP},
        (1, 1): set(),
        (-1, 1): set(),
    }
    ctx = _FakeContext(grid)
    ctx.bot.statistics.visited[(-1, 1)] = 4

    path, steps = _run_macro(MomentumCorridorRight(), ctx)

    assert steps == 2
    assert path == [(0, 0), (0, 1), (1, 1)]


def test_momentum_macro_hands_control_back_after_local_choice_budget() -> None:
    grid = {
        (0, 0): {RIGHT},
        (0, 1): {RIGHT, DOWN},
        (0, 2): {RIGHT, DOWN},
        (0, 3): {RIGHT, DOWN},
        (1, 1): set(),
        (1, 2): set(),
        (1, 3): set(),
    }
    ctx = _FakeContext(grid)
    ctx.bot.statistics.visited[(1, 1)] = 1
    ctx.bot.statistics.visited[(1, 2)] = 1
    ctx.bot.statistics.visited[(1, 3)] = 1

    path, steps = _run_macro(MomentumCorridorRight(), ctx)

    assert steps == 3
    assert path == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_momentum_macro_resets_state_between_runs() -> None:
    grid = {(0, column): {RIGHT} for column in range(5)}
    ctx = _FakeContext(grid)
    macro = MomentumCorridorRight()

    _, first_steps = _run_macro(macro, ctx)
    ctx.position = (0, 0)
    _, second_steps = _run_macro(macro, ctx)

    assert first_steps == 5
    assert second_steps == 5


def test_default_options_expose_distinct_macro_sets() -> None:
    naive = defaultOptions("naive")
    momentum = defaultOptions("momentum")
    astar = defaultOptions("astar")

    assert len(naive) == 4
    assert len(momentum) == 4
    assert len(astar) == 5
    assert [option.name for option in momentum] == [
        "momentum_corridor_right",
        "momentum_corridor_left",
        "momentum_corridor_up",
        "momentum_corridor_down",
    ]
    assert all(isinstance(option, MomentumCorridorMacro) for option in momentum)
    assert not ({option.id for option in naive} & {option.id for option in momentum})
    assert astar[-1].name == "astar_shortest_path_to_next_decision"


@pytest.mark.parametrize("option_set", ["naive", "momentum", "astar"])
def test_dqn_config_accepts_current_macro_option_sets(option_set: str) -> None:
    config = DQNConfig.from_profile_dict({"useMacroActions": True, "macroOptionSet": option_set})

    assert config.useMacroActions is True
    assert config.macroOptionSet == option_set


def test_dqn_config_rejects_unknown_macro_option_set() -> None:
    with pytest.raises(ConfigValidationError):
        DQNConfig.from_profile_dict({"useMacroActions": True, "macroOptionSet": "unknown_set"})
