from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from environment.context import EnvironmentContext
    from environment.entity_registry import EntityRegistry
    from environment.sensing import MazeSensingService


class ObservationEncoder:
    """
    Entity-driven observation encoder.

    Each entity that implements ProvidesObservation contributes a feature dict.
    New entity types extend observation automatically — bots need no changes.

    This encoder produces a flat float32 array describing:
      - Basic spatial features (walls, goal distance)
      - Entity features around the agent (aggregated from entity registry)
    """

    # Feature keys produced by this encoder (in order)
    FEATURE_KEYS: list[str] = [
        "is_blocking",
        "is_dangerous",
        "damage_amount",
        "is_goal",
        "is_enemy",
        "distance_to_agent",
        "can_be_attacked",
        "blocks_line_of_sight",
    ]

    def __init__(self, scale: float = 1.0) -> None:
        self._scale = max(1.0, scale)

    def encode(self, context: EnvironmentContext) -> np.ndarray:
        return self.encodeForPosition(context, context.position)

    def encodeForPosition(
        self,
        context: EnvironmentContext,
        position: tuple[int, int],
    ) -> np.ndarray:
        """
        Returns a flat float32 feature vector for a specific agent position.
        """
        features: list[float] = []

        # Entity features at current cell (aggregated)
        cell_features = self._aggregateEntityFeatures(context, position)
        features.extend(self._featureVector(cell_features))

        # Entity features in each of 4 cardinal directions (first cell only)
        for d in range(4):
            from bots.common.actions import DIRECTION_DELTAS
            dr, dc = DIRECTION_DELTAS[d]
            adj = (position[0] + dr, position[1] + dc)
            adj_features = self._aggregateEntityFeatures(context, adj)
            features.extend(self._featureVector(adj_features))

        # Wall / movement validity in each direction (4 bits) for the provided position.
        for adj in self._adjacentPositions(position):
            features.append(1.0 if context.isValidPosition(adj) else 0.0)

        # Goal info
        goal = context.goalPosition
        scale = self._scale
        features.append(float(np.clip((goal[0] - position[0]) / scale, -1.0, 1.0)))
        features.append(float(np.clip((goal[1] - position[1]) / scale, -1.0, 1.0)))
        features.append(1.0 if position == goal else 0.0)

        return np.array(features, dtype=np.float32)

    def encodeCompactState(
        self,
        context: EnvironmentContext,
        sensing_service: MazeSensingService,
        *,
        position: tuple[int, int] | None = None,
        include_entity_features: bool = False,
        include_enemy_features: bool = False,
    ) -> tuple[Any, ...]:
        pos = context.position if position is None else position
        wall_distances, goal_direction = sensing_service.wall_distances_and_goal_directions(pos)
        entity_features: tuple[float, ...] = ()
        if include_entity_features:
            entity_features = tuple(
                float(v) for v in self.encodeForPosition(context, pos).tolist()
            )
        if not include_enemy_features:
            return pos, wall_distances, goal_direction, entity_features
        enemy_features = self.compactEnemyFeatures(context, sensing_service, position=pos)
        return pos, wall_distances, goal_direction, entity_features, enemy_features

    def encodeEnemyScan(
        self,
        context: EnvironmentContext,
        sensing_service: MazeSensingService,
        *,
        position: tuple[int, int] | None = None,
    ) -> tuple[float, float, float, float]:
        """Returns (dist_norm, dr_norm, dc_norm, in_los) for DQN continuous observation."""
        pos = context.position if position is None else position
        return sensing_service.enemy_scan(pos, context.entityRegistry)

    def compactEnemyFeatures(
        self,
        context: EnvironmentContext,
        sensing_service: MazeSensingService,
        *,
        position: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """
        Returns (dir_bucket, dist_bucket) for Q-learning discrete state.

        dir_bucket: 0=no enemy, 1=up, 2=down, 3=left, 4=right (quadrant toward nearest)
        dist_bucket: 0=no enemy, 1=close (1-3), 2=medium (4-7), 3=far (8+)
        """
        pos = context.position if position is None else position
        dist_norm, dr_norm, dc_norm, _ = sensing_service.enemy_scan(pos, context.entityRegistry)
        if dist_norm == 0.0 and dr_norm == 0.0 and dc_norm == 0.0:
            return (0, 0)
        scale = float(max(int(sensing_service.maze.height), int(sensing_service.maze.width), 1))
        dist_cells = dist_norm * scale
        if dist_cells <= 3:
            dist_bucket = 1
        elif dist_cells <= 7:
            dist_bucket = 2
        else:
            dist_bucket = 3
        if abs(dr_norm) >= abs(dc_norm):
            dir_bucket = 1 if dr_norm < 0 else 2  # up / down
        else:
            dir_bucket = 3 if dc_norm < 0 else 4  # left / right
        return (dir_bucket, dist_bucket)

    def observationDim(self) -> int:
        n_cells = 5   # current + 4 adjacent
        n_entity_features = len(self.FEATURE_KEYS)
        n_direction_bits = 4
        n_goal_info = 3
        return n_cells * n_entity_features + n_direction_bits + n_goal_info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _aggregateEntityFeatures(
        self, context: EnvironmentContext, position: tuple[int, int]
    ) -> dict[str, float]:
        agg: dict[str, float] = {k: 0.0 for k in self.FEATURE_KEYS}
        for feat_dict in context.entityRegistry.observationFeaturesAt(position, context):
            for key, val in feat_dict.items():
                if key in agg:
                    agg[key] = max(agg[key], float(val))
        return agg

    def _featureVector(self, features: dict[str, float]) -> list[float]:
        return [features.get(k, 0.0) for k in self.FEATURE_KEYS]

    @staticmethod
    def _adjacentPositions(position: tuple[int, int]) -> list[tuple[int, int]]:
        from bots.common.actions import DIRECTION_DELTAS

        return [
            (position[0] + dr, position[1] + dc)
            for dr, dc in DIRECTION_DELTAS.values()
        ]
