from __future__ import annotations

from dataclasses import dataclass, field
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
        if self._has_line_of_sight(enemy, context):
            self._set_locked(enemy, True)
        if not self._is_locked(enemy):
            return EnemyAction(direction=None)

        direction = self._shortest_path_direction(enemy, context)
        if direction is not None:
            return EnemyAction(direction=direction)
        return EnemyAction(direction=self._direct_direction(enemy, context))

    def on_resolution(
        self,
        enemy: Enemy,
        action: EnemyAction,
        *,
        moved: bool,
        blocked_reason: str | None = None,
    ) -> None:
        return

    def _is_locked(self, enemy: Enemy) -> bool:
        rawBehavior = getattr(enemy, "_behavior", None)
        return isinstance(rawBehavior, dict) and bool(rawBehavior.get("locked_on", False))

    def _set_locked(self, enemy: Enemy, locked: bool) -> None:
        rawBehavior = getattr(enemy, "_behavior", None)
        behavior = dict(rawBehavior) if isinstance(rawBehavior, dict) else {"kind": "chase"}
        behavior["kind"] = "chase"
        behavior["locked_on"] = bool(locked)
        enemy._behavior = behavior

    def _has_line_of_sight(self, enemy: Enemy, context: EnvironmentContext) -> bool:
        botRow, botCol = context.position
        row, col = enemy.position
        if botRow == row:
            step = 1 if botCol > col else -1
            return all(context.isValidPosition((row, c)) for c in range(col + step, botCol, step))
        if botCol == col:
            step = 1 if botRow > row else -1
            return all(context.isValidPosition((r, col)) for r in range(row + step, botRow, step))
        return False

    def _shortest_path_direction(self, enemy: Enemy, context: EnvironmentContext) -> int | None:
        start = cast(tuple[int, int], enemy.position)
        goal = context.position
        if start == goal:
            return None
        queue: list[tuple[int, int]] = [start]
        previous: dict[tuple[int, int], tuple[tuple[int, int] | None, int | None]] = {start: (None, None)}
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            if current == goal:
                break
            for direction in self._DIRECTION_ORDER:
                dr, dc = DIRECTION_DELTAS[int(direction)]
                nxt = (current[0] + dr, current[1] + dc)
                if nxt in previous:
                    continue
                if nxt != goal and not context.isValidPosition(nxt):
                    continue
                if nxt == goal or context.isValidPosition(nxt):
                    previous[nxt] = (current, int(direction))
                    queue.append(nxt)

        if goal not in previous:
            return None

        node = goal
        firstDirection: int | None = None
        while node != start:
            parent, direction = previous[node]
            firstDirection = direction
            if parent is None:
                break
            node = parent
        return firstDirection

    def _direct_direction(self, enemy: Enemy, context: EnvironmentContext) -> int | None:
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
            return None
        candidates.sort(key=lambda item: (item[1], self._DIRECTION_ORDER.index(item[0])))
        return candidates[0][0]


class EnemyBehaviorFactory:
    @staticmethod
    def for_enemy(enemy: Enemy) -> EnemyBehavior:
        rawBehavior = getattr(enemy, "_behavior", None)
        behaviorKind = "stationary"
        if isinstance(rawBehavior, dict):
            behaviorKind = str(rawBehavior.get("kind", "stationary")).strip().lower()
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
        if any(isinstance(entity, Enemy) and getattr(entity, "_alive", True) for entity in occupants):
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
            if not getattr(entity, "_alive", True):
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
