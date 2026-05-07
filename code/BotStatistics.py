from typing import Dict, Tuple


class BotStatistics:
    """In-memory episode statistics only.

    Persistence and aggregation are handled by services.repository.ArtifactsRepository.
    """
    def __init__(self) -> None:
        self.total_steps: int = 0
        self.times_hit_wall: int = 0
        self.times_revisited_squares: int = 0
        self.non_repeating_steps_taken: int = 0
        self.visited_positions: Dict[Tuple[int, int], int] = {}
        self.last_visited_positions: list[Tuple[int, int]] = []

    def reset(self) -> None:
        self.total_steps = 0
        self.times_hit_wall = 0
        self.times_revisited_squares = 0
        self.non_repeating_steps_taken = 0
        self.visited_positions.clear()
        self.last_visited_positions.clear()

    def update_visited_positions(self, position: Tuple[int, int]) -> None:
        """Update the count of times a position has been visited."""
        self.visited_positions[position] = self.visited_positions.get(position, 0) + 1

    def get_visited_positions(self) -> Dict[Tuple[int, int], int]:
        """Retrieve the dictionary of visited positions."""
        return self.visited_positions

    def update_last_visited(self, position: Tuple[int, int]) -> None:
        """Update the list of the last visited positions."""
        if len(self.last_visited_positions) >= 5:
            self.last_visited_positions.pop(0)
        if position not in self.last_visited_positions:
            self.last_visited_positions.append(position)

    def get_last_visited(self) -> list[Tuple[int, int]]:
        """Retrieve the list of the last visited positions."""
        return self.last_visited_positions
