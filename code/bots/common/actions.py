from __future__ import annotations

from dataclasses import dataclass

# ── Direction constants (canonical action IDs for all bots) ──────────────────
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

NUM_PRIMITIVE_DIRECTIONS = 4

# Attack action IDs — map same direction deltas as moves, IDs 4-7
ATTACK_UP = 4
ATTACK_DOWN = 5
ATTACK_LEFT = 6
ATTACK_RIGHT = 7

DIRECTION_DELTAS: dict[int, tuple[int, int]] = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    # Attack actions share the same directional deltas as movement actions
    ATTACK_UP: (-1, 0),
    ATTACK_DOWN: (1, 0),
    ATTACK_LEFT: (0, -1),
    ATTACK_RIGHT: (0, 1),
}

DIRECTION_NAMES: dict[int, str] = {
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
}

NAME_TO_DIRECTION: dict[str, int] = {v: k for k, v in DIRECTION_NAMES.items()}


@dataclass(frozen=True)
class ActionSpec:
    """Describes one action in the vocabulary — primitive, macro, or option."""
    id: int
    name: str
    kind: str  # "primitive" | "macro" | "option"
    description: str = ""


@dataclass(frozen=True)
class PrimitiveActionSpec(ActionSpec):
    """Primitive action: consumes one environment step, executed via executor_key."""
    executor_key: str = ""


# ── Default primitive movement specs ─────────────────────────────────────────

PRIMITIVE_MOVE_UP = PrimitiveActionSpec(
    id=UP, name="move_up", kind="primitive",
    description="Move one cell upward (row-1)", executor_key="move_up",
)
PRIMITIVE_MOVE_DOWN = PrimitiveActionSpec(
    id=DOWN, name="move_down", kind="primitive",
    description="Move one cell downward (row+1)", executor_key="move_down",
)
PRIMITIVE_MOVE_LEFT = PrimitiveActionSpec(
    id=LEFT, name="move_left", kind="primitive",
    description="Move one cell left (col-1)", executor_key="move_left",
)
PRIMITIVE_MOVE_RIGHT = PrimitiveActionSpec(
    id=RIGHT, name="move_right", kind="primitive",
    description="Move one cell right (col+1)", executor_key="move_right",
)

DEFAULT_PRIMITIVE_ACTIONS: list[PrimitiveActionSpec] = [
    PRIMITIVE_MOVE_UP,
    PRIMITIVE_MOVE_DOWN,
    PRIMITIVE_MOVE_LEFT,
    PRIMITIVE_MOVE_RIGHT,
]

PRIMITIVE_ATTACK_UP = PrimitiveActionSpec(
    id=ATTACK_UP, name="attack_up", kind="primitive",
    description="Attack the cell above (kills enemy in 1 hit)", executor_key="attack_up",
)
PRIMITIVE_ATTACK_DOWN = PrimitiveActionSpec(
    id=ATTACK_DOWN, name="attack_down", kind="primitive",
    description="Attack the cell below (kills enemy in 1 hit)", executor_key="attack_down",
)
PRIMITIVE_ATTACK_LEFT = PrimitiveActionSpec(
    id=ATTACK_LEFT, name="attack_left", kind="primitive",
    description="Attack the cell to the left (kills enemy in 1 hit)", executor_key="attack_left",
)
PRIMITIVE_ATTACK_RIGHT = PrimitiveActionSpec(
    id=ATTACK_RIGHT, name="attack_right", kind="primitive",
    description="Attack the cell to the right (kills enemy in 1 hit)", executor_key="attack_right",
)

DEFAULT_ATTACK_ACTIONS: list[PrimitiveActionSpec] = [
    PRIMITIVE_ATTACK_UP,
    PRIMITIVE_ATTACK_DOWN,
    PRIMITIVE_ATTACK_LEFT,
    PRIMITIVE_ATTACK_RIGHT,
]
