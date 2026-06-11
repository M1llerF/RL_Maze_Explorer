from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from maze import Maze
from pathfinding import Pathfinding
from services.research.plan import ConditionSpec


def _resolve_path(base_dir: str, raw_path: str) -> str:
    path = raw_path if os.path.isabs(raw_path) else os.path.abspath(os.path.join(base_dir, raw_path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Maze file not found: {path}")
    return path


def _file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load_state_from_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Maze file must contain a JSON object: {path}")
    return data


def _maze_from_state(condition: ConditionSpec, state: dict[str, Any], *, source_path: str) -> Maze:
    maze = Maze(condition.width, condition.height)
    maze.setState(state)
    if int(maze.width) != int(condition.width) or int(maze.height) != int(condition.height):
        raise ValueError(
            f"Maze file size mismatch for condition '{condition.condition_id}': "
            f"expected {condition.width}x{condition.height}, got {maze.width}x{maze.height} "
            f"from {source_path}"
        )
    return maze


def load_fixed_maze(condition: ConditionSpec, base_dir: str) -> tuple[Maze, dict[str, Any]]:
    source_path = _resolve_path(base_dir, condition.maze_path)
    state = _load_state_from_json(source_path)
    maze = _maze_from_state(condition, state, source_path=source_path)
    return maze, {
        "source_path": source_path,
        "source_sha256": _file_sha256(source_path),
    }


def load_pool_mazes_from_paths(
    condition: ConditionSpec,
    base_dir: str,
    raw_paths: list[str],
) -> tuple[list[Maze], list[dict[str, Any]]]:
    mazes: list[Maze] = []
    metadata: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        source_path = _resolve_path(base_dir, raw_path)
        state = _load_state_from_json(source_path)
        mazes.append(_maze_from_state(condition, state, source_path=source_path))
        metadata.append({
            "source_path": source_path,
            "source_sha256": _file_sha256(source_path),
        })
    return mazes, metadata


def load_pool_mazes(condition: ConditionSpec, base_dir: str) -> tuple[list[Maze], list[dict[str, Any]]]:
    return load_pool_mazes_from_paths(condition, base_dir, condition.maze_paths)


def compute_maze_stats(maze: Maze) -> dict[str, Any]:
    start = maze.start
    end = maze.end
    optimal_path: list[tuple[int, int]] = []
    if start is not None and end is not None:
        optimal_path = list(Pathfinding.aStarSearch(maze, start, end))
    wall_count = sum(maze.grid[r][c] for r in range(maze.height) for c in range(maze.width))
    total_cells = maze.width * maze.height
    open_count = total_cells - wall_count
    return {
        "start": list(start) if start else [0, 0],
        "end": list(end) if end else [0, 0],
        "grid": [row[:] for row in maze.grid],
        "optimal_path": [list(p) for p in optimal_path],
        "optimal_path_length": len(optimal_path),
        "wall_count": wall_count,
        "open_cell_count": open_count,
        "wall_density": round(wall_count / total_cells, 4) if total_cells > 0 else 0.0,
    }


def export_fixed_maze_json(
    condition: ConditionSpec,
    maze: Maze,
    *,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    stats = compute_maze_stats(maze)
    return {
        "condition": condition.condition_id,
        "mode": "fixed",
        "width": condition.width,
        "height": condition.height,
        "maze_id": condition.maze_id,
        "source_path": source_path,
        "source_sha256": source_sha256,
        **stats,
    }


def export_pool_maze_json(
    condition: ConditionSpec,
    mazes: list[Maze],
    metadata: list[dict[str, Any]],
    *,
    split: str | None = None,
) -> dict[str, Any]:
    maze_entries: list[dict[str, Any]] = []
    for i, maze in enumerate(mazes):
        stats = compute_maze_stats(maze)
        maze_entries.append({
            "maze_id": f"{condition.maze_id_prefix}_{i:02d}",
            "source_path": metadata[i]["source_path"],
            "source_sha256": metadata[i]["source_sha256"],
            **stats,
        })
    return {
        "condition": condition.condition_id,
        "mode": "pool",
        "split": split,
        "width": condition.width,
        "height": condition.height,
        "pool_size": condition.pool_size,
        "mazes": maze_entries,
    }
