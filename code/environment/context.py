from __future__ import annotations

from typing import Any, TYPE_CHECKING

from bots.common.actions import DIRECTION_DELTAS, NAME_TO_DIRECTION
from environment.traversal import TraversalPolicy, build_traversal_policy

if TYPE_CHECKING:
    from environment.entity_registry import EntityRegistry
    from bots.common.action_registry import ActionRegistry

POSITION_CHANGED = "position_changed"
ACTION_APPLIED = "action_applied"
COLLISION = "collision"
GOAL_REACHED = "goal_reached"
ENEMY_MOVED = "enemy_moved"
ENEMY_BLOCKED = "enemy_blocked"
ENEMY_CONTACT = "enemy_contact"
ENEMY_DEFEATED = "enemy_defeated"
ENEMY_KILLED = "enemy_killed"


class EnvironmentContext:
    """
    Single interface through which actions, executors, and options query
    the environment.  Options and executors should never import maze or bot
    classes directly.

    The bot holds one EnvironmentContext instance per episode and passes it to
    ActionRegistry.executePrimitive() and ActionRegistry.executeOption().
    """

    def __init__(
        self,
        maze: Any,
        bot: Any,
        entity_registry: EntityRegistry,
        action_registry: ActionRegistry | None = None,
        traversal_policy: TraversalPolicy | None = None,
    ) -> None:
        self._maze = maze
        self._bot = bot
        self._entities = entity_registry
        self._actions: ActionRegistry | None = action_registry
        self._listeners: dict[str, list[Any]] = {}
        self._traversal = traversal_policy or build_traversal_policy(
            maze,
            getattr(bot, "profileName", None),
        )

    @property
    def bot(self) -> Any:
        return self._bot

    @property
    def maze(self) -> Any:
        return self._maze

    def setActionRegistry(self, registry: ActionRegistry) -> None:
        self._actions = registry

    def on(self, event_name: str, listener: Any) -> None:
        self._listeners.setdefault(str(event_name), []).append(listener)

    def emit(self, event_name: str, **payload: Any) -> None:
        for listener in self._listeners.get(str(event_name), []):
            listener(**payload)

    # ── Position ──────────────────────────────────────────────────────────────

    @property
    def position(self) -> tuple[int, int]:
        return tuple(self._bot.position)  # type: ignore[return-value]

    def moveAgentTo(self, new_position: tuple[int, int]) -> None:
        """Called by movement executors to advance the bot's position."""
        stats = self._bot.statistics
        current = self.position
        visited = stats.getVisitedPositions()

        if new_position in visited:
            stats.timesRevisitedSquares += 1
        else:
            stats.nonRepeatingStepsTaken += 1

        stats.updateLastVisited(current)
        if hasattr(self._bot, "previousPosition"):
            self._bot.previousPosition = current
        self._bot.position = new_position
        stats.updateVisitedPositions(new_position)
        self.emit(
            POSITION_CHANGED,
            previous_position=current,
            new_position=new_position,
            visited_before_move=bool(new_position in visited),
        )

    # ── Movement queries ──────────────────────────────────────────────────────

    def isValidPosition(self, pos: tuple[int, int]) -> bool:
        return bool(self._traversal.is_valid_position(pos))

    def canMove(self, direction: int | str) -> bool:
        from environment.movement import _resolve_direction
        d = _resolve_direction(direction)
        dr, dc = DIRECTION_DELTAS[d]
        target = (self.position[0] + dr, self.position[1] + dc)
        return self.isValidPosition(target) and not self._entities.blocksMovement(target, self, d)

    def isAtJunction(self) -> bool:
        open_dirs = sum(1 for d in range(4) if self.canMove(d))
        return open_dirs >= 3

    def isAtGoal(self) -> bool:
        return self.position == tuple(self._maze.end)

    def isAtDeadEnd(self) -> bool:
        open_dirs = sum(1 for d in range(4) if self.canMove(d))
        return open_dirs <= 1

    # ── Entity queries ────────────────────────────────────────────────────────

    @property
    def entityRegistry(self) -> EntityRegistry:
        return self._entities

    def entitiesAt(self, position: tuple[int, int]) -> list[Any]:
        return self._entities.entitiesAt(position)

    def featuresAroundAgent(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        for entity_features in self._entities.observationFeaturesAt(self.position, self):
            features.update(entity_features)
        return features

    # ── Action registry ───────────────────────────────────────────────────────

    def actionId(self, name: str) -> int:
        if self._actions is None:
            raise RuntimeError("ActionRegistry not set on EnvironmentContext")
        return self._actions.actionId(name)

    # ── Push cooldown (delegated to bot) ─────────────────────────────────────

    @property
    def isPushReady(self) -> bool:
        getter = getattr(self._bot, "isPushReady", None)
        return bool(getter()) if callable(getter) else True

    def consumePushCooldown(self) -> None:
        fn = getattr(self._bot, "consumePushCooldown", None)
        if callable(fn):
            fn()

    def tickPushCooldown(self) -> None:
        fn = getattr(self._bot, "tickPushCooldown", None)
        if callable(fn):
            fn()

    # ── Entity / maze sync ────────────────────────────────────────────────────

    def syncEntitiesToMaze(self) -> None:
        """Rebuild maze.entities from the live entity registry (excludes Wall/Goal)."""
        from environment.entities import entity_to_state, Wall, Goal
        non_structural = [
            entity_to_state(e) for e in self._entities.allEntities()
            if not isinstance(e, (Wall, Goal))
        ]
        try:
            self._maze.entities = non_structural
        except Exception:
            pass

    # ── Maze metadata ─────────────────────────────────────────────────────────

    @property
    def mazeWidth(self) -> int:
        return int(self._maze.width)

    @property
    def mazeHeight(self) -> int:
        return int(self._maze.height)

    @property
    def goalPosition(self) -> tuple[int, int]:
        return tuple(self._maze.end)  # type: ignore[return-value]
