from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Protocol, cast

from bots.common.actions import DIRECTION_DELTAS, UP, DOWN, LEFT, RIGHT

from .context import ENEMY_BLOCKED, ENEMY_CONTACT, ENEMY_DEFEATED, ENEMY_MOVED, EnvironmentContext
from .entities import Enemy, Goal, Wall, entity_to_state


@dataclass(frozen=True)
class EnemyAction:
    direction: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnemyTickResult:
    contact_damage: float = 0.0
    moved_enemy_count: int = 0
    defeated_enemy_ids: tuple[str, ...] = ()
    semantic_events: tuple[dict[str, Any], ...] = ()
    agent_caught: bool = False


@dataclass(frozen=True)
class EnemyOccupancyDecision:
    outcome: str
    target: tuple[int, int]
    blocked_reason: str | None = None
    contact_damage: float = 0.0
    contact_kind: str | None = None
    is_position_swap: bool = False


class EnemyBehavior(Protocol):
    def choose_action(self, enemy: Enemy, context: EnvironmentContext) -> EnemyAction:
        ...

    def on_resolution(
        self,
        enemy: Enemy,
        action: EnemyAction,
        *,
        moved: bool,
        blocked_reason: str | None = None,
    ) -> None:
        ...


class StationaryEnemyBehavior:
    def choose_action(self, enemy: Enemy, context: EnvironmentContext) -> EnemyAction:
        return EnemyAction(direction=None)

    def on_resolution(
        self,
        enemy: Enemy,
        action: EnemyAction,
        *,
        moved: bool,
        blocked_reason: str | None = None,
    ) -> None:
        return


class ChaseEnemyBehavior:
    _DIRECTION_ORDER: tuple[int, int, int, int] = (UP, DOWN, LEFT, RIGHT)

    def choose_action(self, enemy: Enemy, context: EnvironmentContext) -> EnemyAction:
        botRow, botCol = context.position
        row, col = enemy.position
        candidates: list[tuple[int, int]] = []
        if botRow < row:
            candidates.append((UP, abs(botRow - (row - 1)) + abs(botCol - col)))
        elif botRow > row:
            candidates.append((DOWN, abs(botRow - (row + 1)) + abs(botCol - col)))
        if botCol < col:
            candidates.append((LEFT, abs(botRow - row) + abs(botCol - (col - 1))))
        elif botCol > col:
            candidates.append((RIGHT, abs(botRow - row) + abs(botCol - (col + 1))))
        if not candidates:
            return EnemyAction(direction=None)
        candidates.sort(key=lambda item: (item[1], self._DIRECTION_ORDER.index(item[0])))
        return EnemyAction(direction=candidates[0][0])

    def on_resolution(
        self,
        enemy: Enemy,
        action: EnemyAction,
        *,
        moved: bool,
        blocked_reason: str | None = None,
    ) -> None:
        return


class RandomWalkerEnemyBehavior:
    _DIRECTION_ORDER: tuple[int, int, int, int] = (UP, DOWN, LEFT, RIGHT)

    def choose_action(self, enemy: Enemy, context: EnvironmentContext) -> EnemyAction:
        state = _behavior_state(enemy)
        step = int(state.get("step", 0))
        seed = int(state.get("seed", 0))
        current = cast(tuple[int, int], enemy.position)
        candidates: list[int] = []
        for direction in self._DIRECTION_ORDER:
            dr, dc = DIRECTION_DELTAS[direction]
            target = (current[0] + dr, current[1] + dc)
            if context.isValidPosition(target):
                candidates.append(direction)
        if not candidates:
            return EnemyAction(direction=None)
        rng = random.Random(f"{seed}:{enemy.id}:{step}")
        direction = candidates[rng.randrange(len(candidates))]
        return EnemyAction(direction=direction, metadata={"next_step": step + 1})

    def on_resolution(
        self,
        enemy: Enemy,
        action: EnemyAction,
        *,
        moved: bool,
        blocked_reason: str | None = None,
    ) -> None:
        nextStep = action.metadata.get("next_step")
        if nextStep is not None:
            _behavior_state(enemy)["step"] = int(nextStep)


class PatrolEnemyBehavior:
    def choose_action(self, enemy: Enemy, context: EnvironmentContext) -> EnemyAction:
        state = _behavior_state(enemy)
        rawPath = state.get("path", [])
        path = [
            (int(pos[0]), int(pos[1]))
            for pos in rawPath
            if isinstance(pos, (list, tuple)) and len(pos) == 2
        ]
        if len(path) < 2:
            return EnemyAction(direction=None)

        current = cast(tuple[int, int], enemy.position)
        currentIndex = int(state.get("index", 0))
        reverse = bool(state.get("reverse", False))
        if current in path:
            currentIndex = path.index(current)
        step = -1 if reverse else 1
        nextIndex = currentIndex + step
        loop = bool(state.get("loop", True))
        if nextIndex < 0 or nextIndex >= len(path):
            if loop:
                nextIndex = 0 if nextIndex >= len(path) else len(path) - 1
            else:
                reverse = not reverse
                step = -1 if reverse else 1
                nextIndex = currentIndex + step
                if nextIndex < 0 or nextIndex >= len(path):
                    return EnemyAction(direction=None)
        target = path[nextIndex]
        direction = _direction_to_neighbor(current, target)
        if direction is None:
            return EnemyAction(direction=None)
        return EnemyAction(
            direction=direction,
            metadata={"next_index": nextIndex, "reverse": reverse},
        )

    def on_resolution(
        self,
        enemy: Enemy,
        action: EnemyAction,
        *,
        moved: bool,
        blocked_reason: str | None = None,
    ) -> None:
        if not moved:
            return
        state = _behavior_state(enemy)
        if "next_index" in action.metadata:
            state["index"] = int(action.metadata["next_index"])
        if "reverse" in action.metadata:
            state["reverse"] = bool(action.metadata["reverse"])


class EnemyBehaviorFactory:
    @staticmethod
    def for_enemy(enemy: Enemy) -> EnemyBehavior:
        rawBehavior = getattr(enemy, "_behavior", None)
        behaviorKind = "stationary"
        if isinstance(rawBehavior, dict):
            behaviorKind = str(rawBehavior.get("kind", "stationary")).strip().lower()
        if behaviorKind == "random_walk":
            return RandomWalkerEnemyBehavior()
        if behaviorKind == "patrol":
            return PatrolEnemyBehavior()
        if behaviorKind == "chase":
            return ChaseEnemyBehavior()
        return StationaryEnemyBehavior()


class EnemyOccupancyRules:
    def resolve_move(
        self,
        enemy: Enemy,
        context: EnvironmentContext,
        *,
        direction: int,
        target: tuple[int, int],
    ) -> EnemyOccupancyDecision:
        registry = context.entityRegistry
        if target == context.position:
            isPositionSwap = getattr(context.bot, "previousPosition", None) == enemy.position
            return EnemyOccupancyDecision(
                outcome="contact",
                target=target,
                contact_damage=float(enemy.damageAmount(context)),
                contact_kind="passive_contact",
                is_position_swap=bool(isPositionSwap),
            )

        if not context.isValidPosition(target):
            return EnemyOccupancyDecision(
                outcome="blocked",
                target=target,
                blocked_reason="maze_blocked",
            )

        occupants = [
            entity
            for entity in registry.entitiesAt(target)
            if str(entity.id) != str(enemy.id)
        ]
        if any(isinstance(entity, Enemy) and bool(getattr(entity, "_alive", True)) for entity in occupants):
            return EnemyOccupancyDecision(
                outcome="blocked",
                target=target,
                blocked_reason="enemy_overlap",
            )

        if registry.blocksMovement(target, context, int(direction), exclude_entity_id=str(enemy.id)):
            return EnemyOccupancyDecision(
                outcome="blocked",
                target=target,
                blocked_reason="entity_blocked",
            )

        return EnemyOccupancyDecision(outcome="move", target=target)


class EnemySystem:
    """Owns enemy turn resolution on top of the shared entity registry."""

    def __init__(self, occupancy_rules: EnemyOccupancyRules | None = None) -> None:
        self._occupancyRules = occupancy_rules or EnemyOccupancyRules()

    def tick(self, context: EnvironmentContext) -> EnemyTickResult:
        registry = context.entityRegistry
        defeatedIds: list[str] = []
        movedCount = 0
        totalDamage = 0.0
        semanticEvents: list[dict[str, Any]] = []

        for entity in registry.allEntities():
            if not isinstance(entity, Enemy):
                continue
            if not bool(getattr(entity, "_alive", True)):
                defeatedIds.append(str(entity.id))
                continue

            behavior = EnemyBehaviorFactory.for_enemy(entity)
            action = behavior.choose_action(entity, context)
            direction = action.direction
            if direction is None:
                behavior.on_resolution(entity, action, moved=False, blocked_reason="no_action")
                continue

            dr, dc = DIRECTION_DELTAS[int(direction)]
            current = cast(tuple[int, int], entity.position)
            target = (current[0] + dr, current[1] + dc)
            decision = self._occupancyRules.resolve_move(
                entity,
                context,
                direction=int(direction),
                target=target,
            )

            if decision.outcome == "contact":
                totalDamage += float(decision.contact_damage)
                semanticEvents.append({
                    "name": "enemy_contact",
                    "payload": {
                        "enemy_id": str(entity.id),
                        "enemy_position": current,
                        "attempted_position": decision.target,
                        "damage": float(decision.contact_damage),
                        "contact_kind": decision.contact_kind,
                        "is_position_swap": bool(decision.is_position_swap),
                    },
                })
                context.emit(
                    ENEMY_CONTACT,
                    enemy_id=str(entity.id),
                    enemy_position=current,
                    attempted_position=decision.target,
                    damage=float(decision.contact_damage),
                    contact_kind=decision.contact_kind,
                    is_position_swap=bool(decision.is_position_swap),
                )
                behavior.on_resolution(entity, action, moved=False, blocked_reason="contact")
                continue

            if decision.outcome == "blocked":
                context.emit(
                    ENEMY_BLOCKED,
                    enemy_id=str(entity.id),
                    enemy_position=current,
                    attempted_position=decision.target,
                    reason=decision.blocked_reason,
                )
                behavior.on_resolution(
                    entity,
                    action,
                    moved=False,
                    blocked_reason=decision.blocked_reason,
                )
                continue

            registry.move(str(entity.id), decision.target)
            movedCount += 1
            context.emit(
                ENEMY_MOVED,
                enemy_id=str(entity.id),
                previous_position=current,
                new_position=decision.target,
                direction=int(direction),
            )
            behavior.on_resolution(entity, action, moved=True)

        for entityId in defeatedIds:
            registry.remove(entityId)
            context.emit(ENEMY_DEFEATED, enemy_id=entityId)

        self._sync_maze_entities(context)
        return EnemyTickResult(
            contact_damage=totalDamage,
            moved_enemy_count=movedCount,
            defeated_enemy_ids=tuple(defeatedIds),
            semantic_events=tuple(semanticEvents),
            agent_caught=bool(totalDamage > 0.0),
        )

    @staticmethod
    def _sync_maze_entities(context: EnvironmentContext) -> None:
        maze = context.maze
        maze.entities = [
            entity_to_state(entity)
            for entity in context.entityRegistry.allEntities()
            if not isinstance(entity, (Wall, Goal))
        ]


def _behavior_state(enemy: Enemy) -> dict[str, Any]:
    state = getattr(enemy, "_behavior", None)
    if not isinstance(state, dict):
        state = {"kind": "stationary"}
        enemy._behavior = state
    return cast(dict[str, Any], state)


def _direction_to_neighbor(
    current: tuple[int, int],
    target: tuple[int, int],
) -> int | None:
    delta = (target[0] - current[0], target[1] - current[1])
    for direction, directionDelta in DIRECTION_DELTAS.items():
        if tuple(directionDelta) == delta:
            return int(direction)
    return None
