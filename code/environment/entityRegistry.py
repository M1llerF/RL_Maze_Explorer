from __future__ import annotations

from typing import TYPE_CHECKING

from .entities import Entity, BlocksMovement, DamagesAgent, ProvidesObservation

if TYPE_CHECKING:
    from environment.context import EnvironmentContext


class EntityRegistry:
    """
    Holds all active entities in the maze.
    Bots never check entity types directly; they query through this registry.
    """

    def __init__(self) -> None:
        self._by_position: dict[tuple[int, int], list[Entity]] = {}
        self._by_id: dict[str, Entity] = {}

    def register(self, entity: Entity) -> None:
        pos = entity.position
        if pos not in self._by_position:
            self._by_position[pos] = []
        self._by_position[pos].append(entity)
        self._by_id[entity.id] = entity

    def remove(self, entity_id: str) -> None:
        entity = self._by_id.pop(entity_id, None)
        if entity is None:
            return
        bucket = self._by_position.get(entity.position, [])
        self._by_position[entity.position] = [e for e in bucket if e.id != entity_id]
        if not self._by_position[entity.position]:
            self._by_position.pop(entity.position, None)

    def get(self, entity_id: str) -> Entity | None:
        return self._by_id.get(entity_id)

    def allEntities(self) -> list[Entity]:
        return list(self._by_id.values())

    def move(self, entity_id: str, new_position: tuple[int, int]) -> None:
        entity = self._by_id.get(entity_id)
        if entity is None:
            raise KeyError(entity_id)
        oldBucket = self._by_position.get(entity.position, [])
        self._by_position[entity.position] = [e for e in oldBucket if e.id != entity_id]
        if not self._by_position[entity.position]:
            self._by_position.pop(entity.position, None)
        entity.position = new_position
        self._by_position.setdefault(new_position, []).append(entity)

    def entitiesAt(self, position: tuple[int, int]) -> list[Entity]:
        return list(self._by_position.get(position, []))

    def blocksMovement(
        self,
        position: tuple[int, int],
        context: EnvironmentContext,
        direction: int,
        exclude_entity_id: str | None = None,
    ) -> bool:
        for entity in self.entitiesAt(position):
            if exclude_entity_id is not None and str(entity.id) == str(exclude_entity_id):
                continue
            if isinstance(entity, BlocksMovement) and entity.blocksMovement(context, direction):
                return True
        return False

    def totalDamageAt(self, position: tuple[int, int], context: EnvironmentContext) -> float:
        total = 0.0
        for entity in self.entitiesAt(position):
            if isinstance(entity, DamagesAgent):
                total += entity.damageAmount(context)
        return total

    def observationFeaturesAt(
        self, position: tuple[int, int], context: EnvironmentContext
    ) -> list[dict[str, float]]:
        features = []
        for entity in self.entitiesAt(position):
            if isinstance(entity, ProvidesObservation):
                features.append(entity.observationFeatures(context))
        return features

    def clear(self) -> None:
        self._by_position.clear()
        self._by_id.clear()
