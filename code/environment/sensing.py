from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from bots.common.actions import DIRECTION_DELTAS
from environment.traversal import TraversalPolicy, build_traversal_policy

if TYPE_CHECKING:
    from environment.entityRegistry import EntityRegistry


@dataclass
class MazeSensingService:
    maze: Any
    traversal: TraversalPolicy

    def goal_in_sight(self, position: tuple[int, int]) -> int:
        return 1 if any(self.goal_visible_in_direction(position, direction) for direction in range(4)) else 0

    def wall_distances_and_goal_directions(
        self, position: tuple[int, int]
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        distances: list[int] = []
        goal_directions: list[int] = []
        for direction in range(4):
            distance, goal_visible, _ = self.ray_features(position, direction)
            distances.append(distance)
            goal_directions.append(goal_visible)
        return tuple(distances), tuple(goal_directions)  # type: ignore[return-value]

    def ray_features(
        self,
        position: tuple[int, int],
        direction: int,
        *,
        visited: dict[tuple[int, int], int] | None = None,
        max_distance: float | None = None,
    ) -> tuple[int, int, int] | tuple[float, float, float]:
        distance = 0
        goal_visible = 0
        visited_visible = 0
        row, col = position
        drow, dcol = DIRECTION_DELTAS[int(direction)]
        while True:
            row += drow
            col += dcol
            candidate = (row, col)
            if not self._is_within_bounds(candidate):
                break
            if not self.traversal.is_valid_position(candidate):
                break
            distance += 1
            if candidate == tuple(self.maze.end):
                goal_visible = 1
            if visited and visited.get(candidate, 0) > 0:
                visited_visible = 1
        if max_distance is None:
            return distance, goal_visible, visited_visible
        scale = max(float(max_distance), 1.0)
        return (
            float(min(distance / scale, 1.0)),
            float(goal_visible),
            float(visited_visible),
        )

    def visible_open_cells(self, position: tuple[int, int]) -> set[tuple[int, int]]:
        visible: set[tuple[int, int]] = {position}
        for direction in range(4):
            row, col = position
            drow, dcol = DIRECTION_DELTAS[int(direction)]
            while True:
                row += drow
                col += dcol
                candidate = (row, col)
                if not self._is_within_bounds(candidate):
                    break
                if not self.traversal.is_valid_position(candidate):
                    break
                visible.add(candidate)
        return visible

    def first_blocked_cells(self, position: tuple[int, int]) -> set[tuple[int, int]]:
        blocked: set[tuple[int, int]] = set()
        for direction in range(4):
            row, col = position
            drow, dcol = DIRECTION_DELTAS[int(direction)]
            while True:
                row += drow
                col += dcol
                candidate = (row, col)
                if not self._is_within_bounds(candidate):
                    break
                if not self.traversal.is_valid_position(candidate):
                    blocked.add(candidate)
                    break
        return blocked

    def goal_visible_in_direction(self, position: tuple[int, int], direction: int) -> bool:
        _, goal_visible, _ = self.ray_features(position, direction)
        return bool(goal_visible)

    def enemy_scan(
        self,
        position: tuple[int, int],
        entity_registry: "EntityRegistry",
        *,
        scale: float | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Returns (dist_norm, dr_norm, dc_norm, in_los) for the nearest alive enemy.

        dist_norm: Manhattan distance to nearest alive enemy, normalised by scale
                   (0.0 when no enemies exist).
        dr_norm / dc_norm: direction vector components to nearest enemy, clipped
                           to [-1, 1] (both 0.0 when no enemies exist).
        in_los: 1.0 if any alive enemy is visible along an unobstructed cardinal
                ray from position, else 0.0.
        """
        from environment.entities import Enemy

        alive_enemies = [
            e for e in entity_registry.allEntities()
            if isinstance(e, Enemy) and e._alive
        ]

        if not alive_enemies:
            return (0.0, 0.0, 0.0, 0.0)

        row, col = position
        norm_scale = float(scale) if scale is not None else float(
            max(int(self.maze.height), int(self.maze.width), 1)
        )

        # Nearest enemy by Manhattan distance
        def _manhattan(e: Enemy) -> int:
            return abs(e.position[0] - row) + abs(e.position[1] - col)

        nearest = min(alive_enemies, key=_manhattan)
        dist = _manhattan(nearest)
        dist_norm = float(min(dist / norm_scale, 1.0))
        dr_norm = float(max(-1.0, min(1.0, (nearest.position[0] - row) / norm_scale)))
        dc_norm = float(max(-1.0, min(1.0, (nearest.position[1] - col) / norm_scale)))

        # Line-of-sight check: any alive enemy reachable via unobstructed cardinal ray
        enemy_positions = {e.position for e in alive_enemies}
        in_los = 0.0
        for direction in range(4):
            r, c = row, col
            drow, dcol = DIRECTION_DELTAS[direction]
            while True:
                r += drow
                c += dcol
                candidate = (r, c)
                if not self._is_within_bounds(candidate):
                    break
                if not self.traversal.is_valid_position(candidate):
                    break
                if candidate in enemy_positions:
                    in_los = 1.0
                    break

        return (dist_norm, dr_norm, dc_norm, in_los)

    def _is_within_bounds(self, position: tuple[int, int]) -> bool:
        return 0 <= position[0] < int(self.maze.height) and 0 <= position[1] < int(self.maze.width)


def build_sensing_service(maze: Any, profile_name: str | None = None) -> MazeSensingService:
    traversal = build_traversal_policy(maze, profile_name)
    return MazeSensingService(maze=maze, traversal=traversal)
