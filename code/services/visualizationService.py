from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from environment.entities import normalize_enemy_behavior
from services.repository import ArtifactsRepository, HeatmapData


@dataclass(frozen=True)
class HeatmapSnapshot:
    maze: Any
    start: Any
    end: Any
    heatmap_data: HeatmapData


@dataclass(frozen=True)
class EnemyRenderState:
    """Lightweight enemy snapshot for visualization rendering."""
    position: tuple[int, int]
    behavior_kind: str
    alive: bool


@dataclass(frozen=True)
class VisualizationSnapshot:
    profile_name: str
    latest: HeatmapSnapshot | None
    highest: HeatmapSnapshot | None
    lowest: HeatmapSnapshot | None
    q_table: dict[Any, Any] | None
    action_specs: list[Any]
    profile_stats: dict[str, Any]
    reward_path: str
    maze_grid: list[list[int]] | None = None
    maze_start: tuple[int, int] | None = None
    maze_end: tuple[int, int] | None = None
    bot_position: tuple[int, int] | None = None
    path: list[tuple[int, int]] | None = None
    rewards_history: list[float] | None = None
    heatmap_data: HeatmapData | None = None
    enemies: tuple[EnemyRenderState, ...] = ()
    bot_type: str = ""
    policy_mode: str = ""
    bot_status: dict[str, Any] | None = None
    agent_diagnostics: dict[str, Any] | None = None
    bot_known_open: frozenset[tuple[int, int]] | None = None
    bot_observed_enemy_positions: frozenset[tuple[int, int]] | None = None


class VisualizationService:
    def __init__(self, repository: ArtifactsRepository) -> None:
        self._repository = repository

    def build_snapshot(self, profile_name: str, bot: Any) -> VisualizationSnapshot:
        maze_data = self._load_maze_data(profile_name)
        maze = getattr(bot, "maze", None)
        bot_position = self._extract_bot_position(bot)
        enemies = self._extract_enemies(maze)
        return VisualizationSnapshot(
            profile_name=profile_name,
            latest=self._heatmap_snapshot(maze_data.get("latest")),
            highest=self._heatmap_snapshot(maze_data.get("highest")),
            lowest=self._heatmap_snapshot(maze_data.get("lowest")),
            q_table=self._extract_q_table(bot),
            action_specs=self._extract_action_specs(bot),
            profile_stats=dict(self._repository.readProfileStats(profile_name) or {}),
            reward_path=self._repository.rewardsPath(profile_name),
            maze_grid=self._extract_maze_grid(maze),
            maze_start=self._extract_start(maze),
            maze_end=self._extract_end(maze),
            bot_position=bot_position,
            path=self._extract_path(bot),
            rewards_history=self._read_rewards(profile_name),
            heatmap_data=self._extract_heatmap(bot),
            enemies=enemies,
            bot_type=self._extract_bot_type(bot),
            bot_known_open=self._extract_known_open(bot),
            bot_observed_enemy_positions=self._extract_bot_observed_enemy_positions(bot, enemies, bot_position),
            policy_mode=self._extract_policy_mode(bot),
            bot_status=self._extract_bot_status(bot),
            agent_diagnostics=self._extract_agent_diagnostics(bot),
        )

    def _load_maze_data(self, profile_name: str) -> dict[str, Any]:
        return dict(self._repository.loadMazeData(profile_name) or {})

    @staticmethod
    def _heatmap_snapshot(raw: Any) -> HeatmapSnapshot | None:
        if not isinstance(raw, dict):
            return None
        maze = raw.get("maze")
        start = raw.get("start")
        end = raw.get("end")
        heatmap_data = raw.get("heatmap_data")
        if maze is None or start is None or end is None or not isinstance(heatmap_data, dict):
            return None
        return HeatmapSnapshot(
            maze=maze,
            start=start,
            end=end,
            heatmap_data=dict(heatmap_data),
        )

    @staticmethod
    def _extract_q_table(bot: Any) -> dict[Any, Any] | None:
        q_learning = getattr(bot, "qLearning", None)
        q_table = getattr(q_learning, "qTable", None)
        if isinstance(q_table, dict):
            return q_table
        return None

    @staticmethod
    def _extract_action_specs(bot: Any) -> list[Any]:
        getter = getattr(bot, "getPolicyActionSpace", None)
        if callable(getter):
            return list(getter())
        fallback = getattr(bot, "getActionSpace", None)
        if callable(fallback):
            return list(fallback())
        return []

    @staticmethod
    def _extract_maze_grid(maze: Any) -> list[list[int]] | None:
        grid = getattr(maze, "grid", None)
        if isinstance(grid, list):
            return [list(row) for row in grid]
        return None

    @staticmethod
    def _extract_start(maze: Any) -> tuple[int, int] | None:
        if maze is None:
            return None
        getter = getattr(maze, "getStart", None)
        try:
            start = getter() if callable(getter) else getattr(maze, "start", None)
        except Exception:
            start = getattr(maze, "start", None)
        if isinstance(start, tuple) and len(start) == 2:
            return int(start[0]), int(start[1])
        if isinstance(start, list) and len(start) == 2:
            return int(start[0]), int(start[1])
        return None

    @staticmethod
    def _extract_end(maze: Any) -> tuple[int, int] | None:
        end = getattr(maze, "end", None)
        if isinstance(end, tuple) and len(end) == 2:
            return int(end[0]), int(end[1])
        if isinstance(end, list) and len(end) == 2:
            return int(end[0]), int(end[1])
        return None

    @staticmethod
    def _extract_bot_position(bot: Any) -> tuple[int, int] | None:
        position = getattr(bot, "position", None)
        if isinstance(position, tuple) and len(position) == 2:
            return int(position[0]), int(position[1])
        if isinstance(position, list) and len(position) == 2:
            return int(position[0]), int(position[1])
        return None

    @staticmethod
    def _extract_path(bot: Any) -> list[tuple[int, int]] | None:
        candidates = (
            getattr(bot, "_visOptimalPath", None),
            getattr(bot, "_currentOptimalPath", None),
        )
        for candidate in candidates:
            if isinstance(candidate, list):
                path: list[tuple[int, int]] = []
                for cell in candidate:
                    if isinstance(cell, tuple) and len(cell) == 2:
                        path.append((int(cell[0]), int(cell[1])))
                    elif isinstance(cell, list) and len(cell) == 2:
                        path.append((int(cell[0]), int(cell[1])))
                return path
        return None

    @staticmethod
    def _extract_heatmap(bot: Any) -> HeatmapData | None:
        statistics = getattr(bot, "statistics", None)
        getter = getattr(statistics, "getVisitedPositions", None)
        if not callable(getter):
            return None
        try:
            visited = getter()
        except Exception:
            return None
        if not isinstance(visited, dict):
            return None
        return {
            (int(pos[0]), int(pos[1])): int(count)
            for pos, count in visited.items()
            if isinstance(pos, tuple) and len(pos) == 2
        }

    @staticmethod
    def _extract_enemies(maze: Any) -> tuple[EnemyRenderState, ...]:
        """Extract alive and dead enemy states from maze.entities for rendering."""
        raw_entities = getattr(maze, "entities", None)
        if not isinstance(raw_entities, list):
            return ()
        result: list[EnemyRenderState] = []
        for entry in raw_entities:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("type", "")).lower() != "enemy":
                continue
            raw_pos = entry.get("position")
            if not isinstance(raw_pos, (list, tuple)) or len(raw_pos) != 2:
                continue
            position = (int(raw_pos[0]), int(raw_pos[1]))
            kind = str(normalize_enemy_behavior(entry.get("behavior", {"kind": "stationary"})).get("kind", "stationary"))
            alive = bool(entry.get("alive", True))
            result.append(EnemyRenderState(position=position, behavior_kind=kind, alive=alive))
        return tuple(result)

    @staticmethod
    def _extract_known_open(bot: Any) -> frozenset[tuple[int, int]] | None:
        known = getattr(bot, "_knownOpen", None)
        if isinstance(known, set):
            return frozenset(known)
        return None

    @staticmethod
    def _extract_bot_observed_enemy_positions(
        bot: Any,
        enemies: tuple[EnemyRenderState, ...],
        bot_position: tuple[int, int] | None,
    ) -> frozenset[tuple[int, int]] | None:
        if bot_position is None:
            return None
        config = getattr(bot, "config", None)
        observed: set[tuple[int, int]] = set()
        alive_enemies = [enemy for enemy in enemies if enemy.alive]

        if getattr(config, "useEntityObservation", False):
            from bots.common.actions import DIRECTION_DELTAS

            local_cells = {bot_position}
            for dr, dc in DIRECTION_DELTAS.values():
                local_cells.add((bot_position[0] + dr, bot_position[1] + dc))
            observed.update(enemy.position for enemy in alive_enemies if enemy.position in local_cells)

        if getattr(config, "useEnemyObservation", False) and alive_enemies:
            row, col = bot_position
            nearest = min(
                alive_enemies,
                key=lambda enemy: abs(enemy.position[0] - row) + abs(enemy.position[1] - col),
            )
            observed.add(nearest.position)

        return frozenset(observed)

    @staticmethod
    def _extract_bot_type(bot: Any) -> str:
        return str(getattr(getattr(bot, "__class__", None), "__name__", ""))

    @staticmethod
    def _extract_policy_mode(bot: Any) -> str:
        if hasattr(bot, "_policyMode"):
            return str(getattr(bot, "_policyMode", ""))
        if getattr(bot, "qLearning", None) is not None:
            return "tabular"
        return ""

    @staticmethod
    def _extract_bot_status(bot: Any) -> dict[str, Any] | None:
        getter = getattr(bot, "getStatus", None)
        if not callable(getter):
            return None
        try:
            status = getter()
        except Exception:
            return None
        if dataclasses.is_dataclass(status):
            return dataclasses.asdict(status)
        raw = getattr(status, "__dict__", None)
        if isinstance(raw, dict):
            return dict(raw)
        return None

    @staticmethod
    def _extract_agent_diagnostics(bot: Any) -> dict[str, Any] | None:
        agent = getattr(bot, "agent", None)
        diag_fn = getattr(agent, "diagnostics", None)
        if not callable(diag_fn):
            return None
        try:
            diagnostics = diag_fn()
        except Exception:
            return None
        if dataclasses.is_dataclass(diagnostics):
            return dataclasses.asdict(diagnostics)
        raw = getattr(diagnostics, "__dict__", None)
        if isinstance(raw, dict):
            return dict(raw)
        return None

    def _read_rewards(self, profile_name: str) -> list[float]:
        path = self._repository.rewardsPath(profile_name)
        try:
            with open(path, "r") as handle:
                values: list[float] = []
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        values.append(float(stripped))
                    except ValueError:
                        continue
                return values
        except Exception:
            return []
