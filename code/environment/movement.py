from __future__ import annotations

from typing import Any, TYPE_CHECKING

from bots.common.actions import (
    DIRECTION_DELTAS, UP, DOWN, LEFT, RIGHT, NAME_TO_DIRECTION,
    ATTACK_UP, ATTACK_DOWN, ATTACK_LEFT, ATTACK_RIGHT,
)
from bots.common.step_result import StepResult
from environment.context import ACTION_APPLIED, COLLISION, GOAL_REACHED, ENEMY_CONTACT, ENEMY_KILLED, ENEMY_MOVED

if TYPE_CHECKING:
    from environment.context import EnvironmentContext


# ── Abstract movement interface ───────────────────────────────────────────────

class MovementModel:
    """
    Strategy interface for movement behaviour.
    Swap implementations to get slippery floors, weighted terrain,
    enemy-aware movement, etc. without touching bot code.
    """

    def candidatePosition(
        self, position: tuple[int, int], direction: int
    ) -> tuple[int, int]:
        """Return the cell the agent would move to (may differ from simple delta for teleport/slide)."""
        dr, dc = DIRECTION_DELTAS[direction]
        return position[0] + dr, position[1] + dc

    def canMove(self, context: EnvironmentContext, direction: int | str) -> bool:
        raise NotImplementedError

    def applyMove(self, context: EnvironmentContext, direction: int | str) -> StepResult:
        raise NotImplementedError


# ── Standard 4-way grid movement ─────────────────────────────────────────────

class GridMovement4Way(MovementModel):
    """
    Plain grid movement in four cardinal directions.
    Blocked by walls (maze.isValidPosition) and registered BlocksMovement entities.
    """

    def canMove(self, context: EnvironmentContext, direction: int | str) -> bool:
        direction = _resolve_direction(direction)
        target = self.candidatePosition(context.position, direction)
        return (
            context.isValidPosition(target)
            and not context.entityRegistry.blocksMovement(target, context, direction)
        )

    def applyMove(self, context: EnvironmentContext, direction: int | str) -> StepResult:
        direction = _resolve_direction(direction)
        target = self.candidatePosition(context.position, direction)

        # Walking into an alive enemy is lethal for the bot.
        alive_enemy = _aliveEnemyAt(context, target)
        if alive_enemy is not None:
            damage = float(alive_enemy.damageAmount(context))
            context.emit(
                ENEMY_CONTACT,
                enemy_id=alive_enemy.id,
                enemy_position=alive_enemy.position,
                attempted_position=target,
                damage=damage,
                contact_kind="bot_walked_into_enemy",
                is_position_swap=False,
            )
            return StepResult(
                reward=-damage,
                done=True,
                info={"hit_wall": False, "direction": direction, "death_by_enemy": True},
                duration=1,
            )

        # Normal movement (no enemy in target cell)
        if not self.canMove(context, direction):
            context.emit(
                COLLISION,
                position=context.position,
                attempted_position=target,
                direction=direction,
            )
            return StepResult(reward=0.0, done=False, info={"hit_wall": True, "direction": direction}, duration=1)

        context.moveAgentTo(target)
        damage = context.entityRegistry.totalDamageAt(target, context)
        done = context.isAtGoal()
        context.emit(
            ACTION_APPLIED,
            previous_position=(target[0] - DIRECTION_DELTAS[direction][0], target[1] - DIRECTION_DELTAS[direction][1]),
            new_position=target,
            direction=direction,
            damage=damage,
            goal_reached=done,
        )
        if done:
            context.emit(GOAL_REACHED, position=target, direction=direction)
        return StepResult(
            reward=-damage,
            done=done,
            info={"hit_wall": False, "direction": direction, "damage": damage},
            duration=1,
        )


# ── Executor wrapper (called by ActionRegistry) ───────────────────────────────

class MoveActionExecutor:
    """Bridges ActionRegistry → MovementModel for a fixed direction."""

    def __init__(self, direction: int, movement_model: MovementModel) -> None:
        self.direction = direction
        self.movement_model = movement_model

    def isValid(self, context: EnvironmentContext) -> bool:
        return self.movement_model.canMove(context, self.direction)

    def execute(self, context: EnvironmentContext) -> StepResult:
        return self.movement_model.applyMove(context, self.direction)


# ── Factory ───────────────────────────────────────────────────────────────────

def buildDefaultMovementExecutors(
    movement_model: MovementModel | None = None,
) -> dict[int, MoveActionExecutor]:
    """Return {action_id: MoveActionExecutor} for all four cardinal directions."""
    model = movement_model or GridMovement4Way()
    return {
        UP: MoveActionExecutor(UP, model),
        DOWN: MoveActionExecutor(DOWN, model),
        LEFT: MoveActionExecutor(LEFT, model),
        RIGHT: MoveActionExecutor(RIGHT, model),
    }


def _resolve_direction(direction: int | str) -> int:
    if isinstance(direction, str):
        d = NAME_TO_DIRECTION.get(direction.lower())
        if d is None:
            raise ValueError(f"Unknown direction name: {direction!r}")
        return d
    return int(direction)


def _aliveEnemyAt(context: EnvironmentContext, position: tuple[int, int]) -> Any:
    """Return the first alive Enemy at position, or None."""
    from environment.entities import Enemy
    for entity in context.entityRegistry.entitiesAt(position):
        if isinstance(entity, Enemy) and entity._alive:
            return entity
    return None


# ── Attack executor (1-hit kill, no cooldown) ─────────────────────────────────

class AttackActionExecutor:
    """
    Executes a directional attack in place (bot does not move).

    If an alive enemy occupies the adjacent cell:
      - Attempts to push enemy one further cell in the same direction.
      - Push to open space: enemy displaced then killed (1-hit).
      - Push into wall / obstacle: enemy dies in place.
      - Either way, enemy always dies. Bot stays put.
    If no enemy: wasted action (0 reward).
    """

    # Maps attack action IDs → movement direction delta
    _ATTACK_TO_MOVE = {
        ATTACK_UP: UP,
        ATTACK_DOWN: DOWN,
        ATTACK_LEFT: LEFT,
        ATTACK_RIGHT: RIGHT,
    }

    def __init__(self, attack_id: int) -> None:
        self._attack_id = attack_id
        self._direction = self._ATTACK_TO_MOVE[attack_id]

    def isValid(self, context: EnvironmentContext) -> bool:
        dr, dc = DIRECTION_DELTAS[self._direction]
        target = (context.position[0] + dr, context.position[1] + dc)
        return _aliveEnemyAt(context, target) is not None

    def execute(self, context: EnvironmentContext) -> StepResult:
        dr, dc = DIRECTION_DELTAS[self._direction]
        target = (context.position[0] + dr, context.position[1] + dc)
        enemy = _aliveEnemyAt(context, target)

        if enemy is None:
            return StepResult(reward=0.0, done=False, info={"attack_miss": True}, duration=1)

        stats = getattr(context.bot, "statistics", None)
        if stats is not None and hasattr(stats, "timesHitEnemy"):
            stats.timesHitEnemy = int(getattr(stats, "timesHitEnemy", 0)) + 1

        push_target = (target[0] + dr, target[1] + dc)
        can_push = (
            context.isValidPosition(push_target)
            and not context.entityRegistry.blocksMovement(push_target, context, self._direction)
        )

        if can_push:
            # Open space behind enemy — displaced, survives
            context.entityRegistry.move(enemy.id, push_target)
            context.emit(ENEMY_MOVED, enemy_id=enemy.id, new_position=push_target, cause="sword_push")
            context.syncEntitiesToMaze()
            return StepResult(
                reward=0.0,
                done=False,
                info={"attack_hit": True, "enemy_kills": 0},
                duration=1,
            )
        else:
            # Wall behind enemy — dies from impact
            enemy.receiveAttack(context)
            context.emit(ENEMY_KILLED, enemy_id=enemy.id, position=target, cause="sword_push_wall")
            context.syncEntitiesToMaze()
            return StepResult(
                reward=0.0,
                done=False,
                info={"attack_hit": True, "enemy_kills": 1},
                duration=1,
            )


def buildAttackExecutors() -> dict[int, AttackActionExecutor]:
    """Return {attack_action_id: AttackActionExecutor} for all four directions."""
    return {
        ATTACK_UP: AttackActionExecutor(ATTACK_UP),
        ATTACK_DOWN: AttackActionExecutor(ATTACK_DOWN),
        ATTACK_LEFT: AttackActionExecutor(ATTACK_LEFT),
        ATTACK_RIGHT: AttackActionExecutor(ATTACK_RIGHT),
    }
