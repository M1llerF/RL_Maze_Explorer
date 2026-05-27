from __future__ import annotations

from typing import Any, Callable
import ast

class RewardConfig:
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the RewardConfig with default values or provided keyword arguments.
        """
        self.goalReward: int = kwargs.get('goal_reward', 1000)
        self.wallPenalty: int = kwargs.get('wall_penalty', -100)
        self.revisitPenaltyOptimal: int = kwargs.get('revisit_penalty_optimal', -10)
        self.revisitPenaltyNonOptimal: int = kwargs.get('revisit_penalty_non_optimal', -15)
        self.stepPenalty: int = kwargs.get('step_penalty', -1)
        self.goalInSightReward: int = kwargs.get('goal_in_sight_reward', 50)
        self.rewardModifiers: dict[str, str] = kwargs.get('reward_modifiers', {
            'goal_reached': '1000',
            'hit_wall': '-100',
            'revisit_optimal_path': '-10',
            'revisit_non_optimal_path': '-15',
            'move_in_optimal_path': '5',
            'see_goal_new_location': '50',
            'see_goal_revisit': '5',
            'per_move_penalty': '-1'
        })
        # Potential-based reward shaping (progress toward goal), helpful for randomized mazes
        self.usePotentialShaping: bool = kwargs.get('use_potential_shaping', False)
        self.progressScale: float = kwargs.get('progress_scale', 5.0)

    def updateFromDict(self, configDict: dict[str, Any]) -> None:
        """
        Update the attributes of RewardConfig from a dictionary.
        """
        for key, value in configDict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Also update the corresponding reward modifier if applicable
                if key in self.rewardModifiers:
                    self.rewardModifiers[key] = str(value)

    def getModifier(self, key: str, default: str = "0") -> str:
        return self.rewardModifiers.get(key, default)
class MazeSensors:
    """Minimal sensor interface used by RewardSystem.

    Provides goal line-of-sight using only the maze grid.
    """
    def __init__(self, maze: Any) -> None:
        self.maze = maze

    def goalInSight(self, pos: tuple[int, int]) -> int:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            cy, cx = pos
            while True:
                ny, nx = cy + dx, cx + dy
                if 0 <= ny < self.maze.height and 0 <= nx < self.maze.width and self.maze.grid[ny][nx] == 0:
                    if (ny, nx) == self.maze.end:
                        return 1
                    cy, cx = ny, nx
                else:
                    break
        return 0


class RewardSystem:
    def __init__(self, maze: Any, rewardConfig: RewardConfig, sensors: MazeSensors | None = None) -> None:
        self.maze = maze
        self.rewardConfig = rewardConfig
        self.cumulativeReward = 0
        self.timesHitWall = 0
        self.timesRevisitedSquare = 0
        self.nonRepeatingStepsTaken = 0
        self.sensors = sensors or MazeSensors(maze)
    
    def evaluateExpression(self, expression: Any, **kwargs: Any) -> float:
        """
        Safely evaluate a numeric expression.
        Accepts numbers or simple arithmetic strings. Disallows names/calls.
        """
        # Fast-path numerics
        if isinstance(expression, (int, float)):
            return float(expression)
        s = str(expression).strip()
        # Simple direct parse
        try:
            return float(s)
        except Exception:
            pass
        # Safe AST evaluation for +,-,*,/ and parentheses
        try:
            node = ast.parse(s, mode='eval')
            return float(self._evalAst(node.body))
        except Exception:
            # Fallback to zero on invalid input
            return 0.0

    def _evalAst(self, node: Any) -> float:
        binOps: dict[type[Any], Callable[[float, float], float]] = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
        }
        unaryOps: dict[type[Any], Callable[[float], float]] = {
            ast.USub: lambda a: -a,
            ast.UAdd: lambda a: +a,
        }
        if isinstance(node, ast.Constant):  # py>=3.8
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Non-numeric constant")
        if isinstance(node, ast.UnaryOp):
            fn = unaryOps.get(type(node.op))
            if fn is None:
                raise ValueError("Unsupported unary operator")
            return fn(self._evalAst(node.operand))
        if isinstance(node, ast.BinOp):
            fn2 = binOps.get(type(node.op))
            if fn2 is None:
                raise ValueError("Unsupported binary operator")
            return fn2(self._evalAst(node.left), self._evalAst(node.right))
        raise ValueError("Unsupported expression")

    def getReward(
        self,
        prevPosition: tuple[int, int],
        newPosition: tuple[int, int],
        optimalPath: list[tuple[int, int]],
        optimalLength: int,
        visitedPositions: dict[tuple[int, int], int],
    ) -> float:
        """
        Calculate the reward for moving to a new position.
        
        :param prev_position: The previous position of the bot.
        :param new_position: The new position of the bot.
        :param optimal_path: The optimal path to the goal.
        :param optimal_length: The length of the optimal path.
        :param visited_positions: The dictionary of visited positions.
        :return: The calculated reward.
        """
        reward = 0

        context = {
            'optimal_length': optimalLength,
            'visited_positions': visitedPositions,
            'optimal_path': optimalPath,
            'new_position': newPosition
        }


        for key, expr in self.rewardConfig.rewardModifiers.items():
            # Use configured values directly (no scaling by path length)
            valueExpr = str(expr)

            if key == 'goal_reached' and newPosition == self.maze.end:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'hit_wall' and not self.maze.isValidPosition(None, *newPosition):
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'revisit_optimal_path' and newPosition in visitedPositions and newPosition in optimalPath:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'revisit_non_optimal_path' and newPosition in visitedPositions and newPosition not in optimalPath:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'move_in_optimal_path' and newPosition in optimalPath:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'see_goal_new_location' and self.sensors.goalInSight(newPosition) and newPosition not in visitedPositions:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'see_goal_revisit' and self.sensors.goalInSight(newPosition) and newPosition in visitedPositions:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'per_move_penalty':
                reward += self.evaluateExpression(valueExpr, **context)
        
        # Potential-based shaping: reward progress toward goal (distance reduction)
        if self.rewardConfig.usePotentialShaping:
            # Manhattan distance tends to be stable in grid mazes
            def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
                return abs(a[0]-b[0]) + abs(a[1]-b[1])
            prevD = manhattan(prevPosition, self.maze.end)
            newD = manhattan(newPosition, self.maze.end)
            progress = prevD - newD  # positive if closer
            reward += self.rewardConfig.progressScale * progress

        return reward

    def updateRewards(self, reward: int) -> None:
        """
        Update cumulative rewards and other statistics.
        
        :param reward: The reward to update.
        """
        self.cumulativeReward += reward
        if reward == self.evaluateExpression(self.rewardConfig.getModifier('hit_wall')):
            self.timesHitWall += 1
        elif reward == self.evaluateExpression(self.rewardConfig.getModifier('revisit_non_optimal_path')):
            self.timesRevisitedSquare += 1
        else:
            self.nonRepeatingStepsTaken += 1

    def resetRewards(self) -> None:
        """
        Reset rewards and other statistics at the start of each episode.
        """
        self.cumulativeReward = 0
        self.timesHitWall = 0
        self.timesRevisitedSquare = 0
        self.nonRepeatingStepsTaken = 0
