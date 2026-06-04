from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from environment.context import EnvironmentContext


# ── Entity base ───────────────────────────────────────────────────────────────

class Entity:
    """Base for all maze objects (walls, enemies, traps, doors, etc.)."""

    def __init__(self, entity_id: str, entity_type: str, position: tuple[int, int]) -> None:
        self.id = entity_id
        self.type = entity_type
        self.position = position


# ── Behaviour interfaces (mix into Entity subclasses) ─────────────────────────

class BlocksMovement:
    def blocksMovement(self, context: EnvironmentContext, direction: int) -> bool:
        raise NotImplementedError


class DamagesAgent:
    def damageAmount(self, context: EnvironmentContext) -> float:
        raise NotImplementedError


class CanBeAttacked:
    def receiveAttack(self, context: EnvironmentContext) -> None:
        raise NotImplementedError


class ProvidesObservation:
    def observationFeatures(self, context: EnvironmentContext) -> dict[str, float]:
        raise NotImplementedError


# ── Concrete built-in entity types ────────────────────────────────────────────

class Wall(Entity, BlocksMovement):
    def __init__(self, position: tuple[int, int]) -> None:
        super().__init__(entity_id=f"wall_{position}", entity_type="wall", position=position)

    def blocksMovement(self, context: EnvironmentContext, direction: int) -> bool:
        return True

    def observationFeatures(self, context: EnvironmentContext) -> dict[str, float]:
        return {"is_blocking": 1.0, "is_dangerous": 0.0, "is_goal": 0.0, "is_enemy": 0.0}


class Goal(Entity, ProvidesObservation):
    def __init__(self, position: tuple[int, int]) -> None:
        super().__init__(entity_id="goal", entity_type="goal", position=position)

    def observationFeatures(self, context: EnvironmentContext) -> dict[str, float]:
        pos = context.position
        distance = abs(pos[0] - self.position[0]) + abs(pos[1] - self.position[1])
        return {"is_blocking": 0.0, "is_dangerous": 0.0, "is_goal": 1.0, "is_enemy": 0.0,
                "distance_to_agent": float(distance)}


class SpikeTrap(Entity, DamagesAgent, ProvidesObservation):
    def __init__(self, position: tuple[int, int], damage: float = 10.0) -> None:
        super().__init__(entity_id=f"spike_{position}", entity_type="spike_trap", position=position)
        self._damage = damage

    def damageAmount(self, context: EnvironmentContext) -> float:
        return self._damage

    def observationFeatures(self, context: EnvironmentContext) -> dict[str, float]:
        return {"is_blocking": 0.0, "is_dangerous": 1.0, "damage_amount": self._damage,
                "is_goal": 0.0, "is_enemy": 0.0}


class Enemy(Entity, BlocksMovement, DamagesAgent, CanBeAttacked, ProvidesObservation):
    def __init__(
        self,
        position: tuple[int, int],
        damage: float = 20.0,
        behavior: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(entity_id=f"enemy_{position}", entity_type="enemy", position=position)
        self._damage = damage
        self._alive = True
        self._behavior = dict(behavior or {"kind": "stationary"})

    def blocksMovement(self, context: EnvironmentContext, direction: int) -> bool:
        return self._alive

    def damageAmount(self, context: EnvironmentContext) -> float:
        return self._damage if self._alive else 0.0

    def receiveAttack(self, context: EnvironmentContext) -> None:
        self._alive = False

    def observationFeatures(self, context: EnvironmentContext) -> dict[str, float]:
        pos = context.position
        distance = abs(pos[0] - self.position[0]) + abs(pos[1] - self.position[1])
        return {"is_blocking": 1.0 if self._alive else 0.0, "is_dangerous": 1.0 if self._alive else 0.0,
                "damage_amount": self._damage, "is_goal": 0.0, "is_enemy": 1.0,
                "distance_to_agent": float(distance), "can_be_attacked": 1.0}


class LockedDoor(Entity, BlocksMovement, ProvidesObservation):
    def __init__(self, position: tuple[int, int]) -> None:
        super().__init__(entity_id=f"door_{position}", entity_type="locked_door", position=position)
        self._locked = True

    def blocksMovement(self, context: EnvironmentContext, direction: int) -> bool:
        return self._locked

    def unlock(self) -> None:
        self._locked = False

    def observationFeatures(self, context: EnvironmentContext) -> dict[str, float]:
        return {"is_blocking": 1.0 if self._locked else 0.0, "is_dangerous": 0.0,
                "is_goal": 0.0, "is_enemy": 0.0, "can_be_attacked": 0.0}


def entity_to_state(entity: Entity) -> dict[str, Any]:
    state: dict[str, Any] = {
        "type": str(entity.type),
        "position": [int(entity.position[0]), int(entity.position[1])],
    }
    if isinstance(entity, SpikeTrap):
        state["damage"] = float(entity._damage)
    elif isinstance(entity, Enemy):
        state["damage"] = float(entity._damage)
        state["alive"] = bool(entity._alive)
        state["behavior"] = dict(getattr(entity, "_behavior", {"kind": "stationary"}))
    elif isinstance(entity, LockedDoor):
        state["locked"] = bool(entity._locked)
    return state


def entity_from_state(data: dict[str, Any]) -> Entity:
    entityType = str(data.get("type", "")).strip().lower()
    rawPosition = data.get("position")
    if not isinstance(rawPosition, (list, tuple)) or len(rawPosition) != 2:
        raise ValueError(f"Invalid entity position: {rawPosition}")
    position = (int(rawPosition[0]), int(rawPosition[1]))

    if entityType == "spike_trap":
        return SpikeTrap(position, damage=float(data.get("damage", 10.0)))
    if entityType == "enemy":
        enemy = Enemy(
            position,
            damage=float(data.get("damage", 20.0)),
            behavior=cast(dict[str, Any], data.get("behavior", {"kind": "stationary"}))
            if isinstance(data.get("behavior", {"kind": "stationary"}), dict)
            else {"kind": "stationary"},
        )
        if not bool(data.get("alive", True)):
            enemy._alive = False
        return enemy
    if entityType == "locked_door":
        door = LockedDoor(position)
        if not bool(data.get("locked", True)):
            door.unlock()
        return door
    raise ValueError(f"Unsupported entity type: {entityType}")
