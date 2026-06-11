from __future__ import annotations

import random

from maze import Maze
from pathfinding import Pathfinding


def _open_ratio(maze: Maze) -> float:
    open_cells = sum(1 for row in maze.grid for cell in row if cell == 0)
    total_cells = max(1, maze.width * maze.height)
    return float(open_cells) / float(total_cells)


def test_generated_mazes_keep_reasonable_open_area() -> None:
    for width, height in ((10, 10), (15, 15), (20, 20), (31, 31)):
        maze = Maze(width, height)
        assert _open_ratio(maze) >= 0.30


def test_generated_maze_start_and_goal_remain_reachable() -> None:
    maze = Maze(20, 20)

    assert maze.start is not None
    assert maze.end is not None
    assert maze.isValidPosition(None, maze.start[0], maze.start[1])
    assert maze.isValidPosition(None, maze.end[0], maze.end[1])


def test_generated_end_uses_farthest_shortest_path_cell() -> None:
    maze = Maze(5, 3)
    maze.grid = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    maze.start = (0, 0)
    candidates = [
        (row_index, col_index)
        for row_index, row in enumerate(maze.grid)
        for col_index, cell in enumerate(row)
        if cell == 0
    ]

    assert maze.getFarthestValidEndPosition(candidates) == (2, 0)


def test_random_enemy_generation_is_opt_in_and_path_biased() -> None:
    random.seed(7)
    maze = Maze(15, 15)
    maze.setupSimpleMaze(includeEnemies=False)

    assert maze.entities == []

    maze.setupSimpleMaze(includeEnemies=True)

    assert maze.entities
    assert maze.start is not None
    assert maze.end is not None

    optimal_path = set(Pathfinding.aStarSearch(maze, maze.start, maze.end))
    on_path_count = 0
    for entity in maze.entities:
        position = tuple(entity["position"])
        assert entity["type"] == "enemy"
        assert entity["behavior"]["kind"] == "chase"
        assert position != maze.start
        assert position != maze.end
        assert maze.isValidPosition(None, position[0], position[1])
        if position in optimal_path:
            on_path_count += 1

    assert on_path_count >= max(1, len(maze.entities) // 2)
