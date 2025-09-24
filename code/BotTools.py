from Pathfinding import Pathfinding
import numpy as np
from typing import Tuple, List, Union
class BotTools:
    def __init__(self, maze):
        """
        Initialize the BotTools with a given maze.

        :param maze: The maze instance the tools will operate on.
        """
        self.maze = maze
    
    def check_goal_in_sight(self, position):
        """
        Check if the goal is in sight from the given position.

        :param position: The current position of the bot.
        :return: 1 if the goal is in sight, otherwise 0.
        """
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            current_position = position
            while True:
                new_position = (current_position[0] + dx, current_position[1] + dy)
                if self.maze.is_valid_position(None, new_position[0], new_position[1]):
                    if new_position == self.maze.end:
                        return 1 # True
                    current_position = new_position
                else:
                    break
        return 0 # False

    
    def pos_to_state(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert the position to a state index.

        :param position: The current position of the bot.
        :return: The state index corresponding to the position.
        """
        return position
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if the position is valid in the maze.

        :param position: The position to check.
        :return: True if the position is valid, otherwise False.
        """
        return self.maze.is_valid_position(None, position[0], position[1])

    def detect_walls(self, position: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Detect the distance to walls in all directions and check for the goal direction.

        :param position: The current position of the bot.
        :return: A tuple containing distances to walls and goal directions for all four directions.
        """
        directions = {
            'Up': (-1, 0),
            'Down': (1, 0),
            'Left': (0, -1),
            'Right': (0, 1),
        }

        wall_distances = []
        goal_directions = []

        for direction, (dx, dy) in directions.items():
            distance, goal = self._detect_wall_in_direction(position, dx, dy)
            wall_distances.append(distance)
            goal_directions.append(goal)

        return tuple(wall_distances), tuple(goal_directions)
    
    def _detect_wall_in_direction(self, position: Tuple[int, int], dx: int, dy: int) -> Tuple[int, int]:
        """
        Detect the distance to a wall and if the goal is in sight in a specific direction.

        :param position: The current position of the bot.
        :param dx: The change in x-direction.
        :param dy: The change in y-direction.
        :return: A tuple containing the distance to the wall and whether the goal is in sight.
        """
        distance = 0
        goal = 0
        current_position = position

        while True:
            next_position = (current_position[0] + dx, current_position[1] + dy)

            if self._is_within_bounds(next_position) and self.maze.is_valid_position(None, next_position[0], next_position[1]):
                current_position = next_position
                distance += 1
                if next_position == self.maze.end:
                    goal = 1
                    break
            else:
                break

        return distance, goal

    def _is_within_bounds(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is within the maze boundaries.

        :param position: The position to check.
        :return: True if the position is within bounds, otherwise False.
        """
        return 0 <= position[0] < self.maze.height and 0 <= position[1] < self.maze.width

    def get_distance_to_goal(self, position: Tuple[int, int]) -> float:
        """
        Get the Euclidean distance to the goal.

        :param position: The current position of the bot.
        :return: The distance to the goal.
        """
        return np.linalg.norm(np.array(position) - np.array(self.maze.end))
    
    def calculate_next_position(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Calculate the next position based on the current action.

        :param position: The current position of the bot.
        :param action: The action to be taken.
        :return: The next position of the bot.
        """
        direction_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        direction = direction_map[action]
        return position[0] + direction[0], position[1] + direction[1]
    
    def get_optimal_path_info(self, start: Tuple[int, int], end: Tuple[int, int], output: str = 'path') -> Union[List[Tuple[int, int]], int]:
        """
        Get the optimal path or its length from start to end based on the output parameter.

        :param start: The start position.
        :param end: The end position.
        :param output: The type of output ('path' or 'length').
        :return: The optimal path or its length.
        """
        optimal_path = Pathfinding.a_star_search(self.maze, start, end)
        if output == 'path':
            return optimal_path
        elif output == 'length':
            return len(optimal_path)
        else:
            raise ValueError("Output parameter must be 'path' or 'length'")
        