from pathfinding import Pathfinding
import numpy as np
from typing import Any, Tuple, List, Union, cast
from environment.sensing import build_sensing_service
from environment.traversal import build_traversal_policy
from bots.common.actions import DIRECTION_DELTAS


class BotTools:
    def __init__(self, maze: Any, profileName: str | None = None) -> None:
        """
        Initialize the BotTools with a given maze.

        :param maze: The maze instance the tools will operate on.
        """
        self.maze = maze
        self.traversal = build_traversal_policy(maze, profileName)
        self.sensing = build_sensing_service(maze, profileName)
    
    def checkGoalInSight(self, position: Tuple[int, int]) -> int:
        """
        Check if the goal is in sight from the given position.

        :param position: The current position of the bot.
        :return: 1 if the goal is in sight, otherwise 0.
        """
        return self.sensing.goal_in_sight(position)

    
    def posToState(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert the position to a state index.

        :param position: The current position of the bot.
        :return: The state index corresponding to the position.
        """
        return position
    
    def isValidPosition(self, position: Tuple[int, int]) -> bool:
        """
        Check if the position is valid in the maze.

        :param position: The position to check.
        :return: True if the position is valid, otherwise False.
        """
        return self.traversal.is_valid_position(position)

    def detectWalls(self, position: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Detect the distance to walls in all directions and check for the goal direction.

        :param position: The current position of the bot.
        :return: A tuple containing distances to walls and goal directions for all four directions.
        """
        return self.sensing.wall_distances_and_goal_directions(position)
    
    def _detectWallInDirection(self, position: Tuple[int, int], dx: int, dy: int) -> Tuple[int, int]:
        """
        Detect the distance to a wall and if the goal is in sight in a specific direction.

        :param position: The current position of the bot.
        :param dx: The change in x-direction.
        :param dy: The change in y-direction.
        :return: A tuple containing the distance to the wall and whether the goal is in sight.
        """
        distance = 0
        goal = 0
        currentPosition = position

        while True:
            nextPosition = (currentPosition[0] + dx, currentPosition[1] + dy)

            if self._isWithinBounds(nextPosition) and self.traversal.is_valid_position(nextPosition):
                currentPosition = nextPosition
                distance += 1
                if nextPosition == self.maze.end:
                    goal = 1
                    break
            else:
                break

        return distance, goal

    def _isWithinBounds(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is within the maze boundaries.

        :param position: The position to check.
        :return: True if the position is within bounds, otherwise False.
        """
        return 0 <= position[0] < self.maze.height and 0 <= position[1] < self.maze.width

    def getDistanceToGoal(self, position: Tuple[int, int]) -> float:
        """
        Get the Euclidean distance to the goal.

        :param position: The current position of the bot.
        :return: The distance to the goal.
        """
        return float(np.linalg.norm(np.array(position) - np.array(self.maze.end)))
    
    def calculateNextPosition(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Calculate the next position based on the current action.

        :param position: The current position of the bot.
        :param action: The action to be taken.
        :return: The next position of the bot.
        """
        dr, dc = DIRECTION_DELTAS[action]
        return position[0] + dr, position[1] + dc
    
    def getOptimalPathInfo(self, start: Tuple[int, int], end: Tuple[int, int], output: str = 'path') -> Union[List[Tuple[int, int]], int]:
        """
        Get the optimal path or its length from start to end based on the output parameter.

        :param start: The start position.
        :param end: The end position.
        :param output: The type of output ('path' or 'length').
        :return: The optimal path or its length.
        """
        optimalPath = Pathfinding.aStarSearch(self.maze, start, end, traversal=self.traversal)
        if output == 'path':
            return optimalPath
        elif output == 'length':
            return len(optimalPath)
        else:
            raise ValueError("Output parameter must be 'path' or 'length'")

    def getDynamicPathInfo(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        knownWalls: set[Tuple[int, int]] | None = None,
        output: str = 'path',
    ) -> Union[List[Tuple[int, int]], int]:
        dynamicPath = Pathfinding.dStarLiteSearch(self.maze, start, end, knownWalls=set(knownWalls or set()))
        if output == 'path':
            return dynamicPath
        elif output == 'length':
            return len(dynamicPath)
        else:
            raise ValueError("Output parameter must be 'path' or 'length'")

    def getDynamicWarmupActions(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        maxSteps: int | None = None,
        knownOpen: set[Tuple[int, int]] | None = None,
        knownWalls: set[Tuple[int, int]] | None = None,
        seenGoals: set[Tuple[int, int]] | None = None,
    ) -> List[int]:
        return Pathfinding.dStarLiteFrontierWarmupActions(
            self.maze,
            start,
            end,
            maxSteps=maxSteps,
            knownOpen=knownOpen,
            knownWalls=knownWalls,
            seenGoals=seenGoals,
        )
