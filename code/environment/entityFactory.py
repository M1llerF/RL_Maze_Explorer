from __future__ import annotations

from typing import Any

from .entities import Goal, Wall, entity_from_state
from .entityRegistry import EntityRegistry


def build_entity_registry(maze: Any) -> EntityRegistry:
    registry = EntityRegistry()

    goal = getattr(maze, "end", None)
    if goal is not None:
        registry.register(Goal((int(goal[0]), int(goal[1]))))

    for row in range(int(getattr(maze, "height", 0))):
        for col in range(int(getattr(maze, "width", 0))):
            isOpen = bool(maze.isValidPosition(None, row, col))
            if not isOpen:
                registry.register(Wall((row, col)))

    for rawEntity in list(getattr(maze, "entities", []) or []):
        if not isinstance(rawEntity, dict):
            continue
        try:
            registry.register(entity_from_state(rawEntity))
        except (KeyError, ValueError, TypeError) as exc:
            print(f"[EntityFactory] Skipping unrecognized entity {rawEntity!r}: {exc}")

    return registry
