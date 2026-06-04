from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class TraversalPolicy:
    """Read-only traversal contract for grid validation."""

    def is_valid_position(self, position: tuple[int, int]) -> bool:
        raise NotImplementedError


@dataclass
class MazeTraversalPolicy(TraversalPolicy):
    maze: Any
    profile_name: str | None = None

    def is_valid_position(self, position: tuple[int, int]) -> bool:
        row, col = position
        return bool(self.maze.isValidPosition(self.profile_name, row, col))


def build_traversal_policy(maze: Any, profile_name: str | None = None) -> MazeTraversalPolicy:
    return MazeTraversalPolicy(maze=maze, profile_name=profile_name)
