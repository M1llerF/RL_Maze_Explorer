
from typing import Any, Dict, Tuple
from typing import Dict
import ast

class RewardConfig:
    def __init__(self, **kwargs):
        """
        Initialize the RewardConfig with default values or provided keyword arguments.
        """
        self.goal_reward: int = kwargs.get('goal_reward', 1000)
        self.wall_penalty: int = kwargs.get('wall_penalty', -100)
        self.revisit_penalty_optimal: int = kwargs.get('revisit_penalty_optimal', -10)
        self.revisit_penalty_non_optimal: int = kwargs.get('revisit_penalty_non_optimal', -15)
        self.step_penalty: int = kwargs.get('step_penalty', -1)
        self.goal_in_sight_reward: int = kwargs.get('goal_in_sight_reward', 50)
        self.reward_modifiers: Dict[str, str] = kwargs.get('reward_modifiers', {
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
        self.use_potential_shaping: bool = kwargs.get('use_potential_shaping', False)
        self.progress_scale: float = kwargs.get('progress_scale', 5.0)

    def update_from_dict(self, config_dict):
        """
        Update the attributes of RewardConfig from a dictionary.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Also update the corresponding reward modifier if applicable
                if key in self.reward_modifiers:
                    self.reward_modifiers[key] = str(value)
class MazeSensors:
    """Minimal sensor interface used by RewardSystem.

    Provides goal line-of-sight using only the maze grid.
    """
    def __init__(self, maze):
        self.maze = maze

    def goal_in_sight(self, pos: Tuple[int, int]) -> int:
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
    def __init__(self, maze, reward_config, sensors: MazeSensors | None = None):
        self.maze = maze
        self.reward_config = reward_config
        self.cumulative_reward = 0
        self.times_hit_wall = 0
        self.times_revisited_square = 0
        self.non_repeating_steps_taken = 0
        self.sensors = sensors or MazeSensors(maze)
    
    def evaluate_expression(self, expression: Any, **kwargs: Any) -> float:
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
            return float(self._eval_ast(node.body))
        except Exception:
            # Fallback to zero on invalid input
            return 0.0

    def _eval_ast(self, node) -> float:
        import operator as op
        ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Mod: op.mod,
            ast.USub: op.neg,
            ast.UAdd: op.pos,
        }
        if isinstance(node, ast.Num):  # py<3.8
            return float(node.n)
        if isinstance(node, ast.Constant):  # py>=3.8
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Non-numeric constant")
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](self._eval_ast(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](self._eval_ast(node.left), self._eval_ast(node.right))
        raise ValueError("Unsupported expression")

    def get_reward(self, prev_position: Tuple[int, int], new_position: Tuple[int, int], optimal_path: list, optimal_length: int, visited_positions: Dict[Tuple[int, int], int]) -> int:
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
            'optimal_length': optimal_length,
            'visited_positions': visited_positions,
            'optimal_path': optimal_path,
            'new_position': new_position
        }


        for key, expr in self.reward_config.reward_modifiers.items():
            # Use configured values directly (no scaling by path length)
            value_expr = str(expr)

            if key == 'goal_reached' and new_position == self.maze.end:
                reward += self.evaluate_expression(value_expr, **context)
            elif key == 'hit_wall' and not self.maze.is_valid_position(None, *new_position):
                reward += self.evaluate_expression(value_expr, **context)
            elif key == 'revisit_optimal_path' and new_position in visited_positions and new_position in optimal_path:
                reward += self.evaluate_expression(value_expr, **context)
            elif key == 'revisit_non_optimal_path' and new_position in visited_positions and new_position not in optimal_path:
                reward += self.evaluate_expression(value_expr, **context)
            elif key == 'move_in_optimal_path' and new_position in optimal_path:
                reward += self.evaluate_expression(value_expr, **context)
            elif key == 'see_goal_new_location' and self.sensors.goal_in_sight(new_position) and new_position not in visited_positions:
                reward += self.evaluate_expression(value_expr, **context)
            elif key == 'see_goal_revisit' and self.sensors.goal_in_sight(new_position) and new_position in visited_positions:
                reward += self.evaluate_expression(value_expr, **context)
            elif key == 'per_move_penalty':
                reward += self.evaluate_expression(value_expr, **context)
        
        # Potential-based shaping: reward progress toward goal (distance reduction)
        if self.reward_config.use_potential_shaping:
            # Manhattan distance tends to be stable in grid mazes
            def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
                return abs(a[0]-b[0]) + abs(a[1]-b[1])
            prev_d = manhattan(prev_position, self.maze.end)
            new_d = manhattan(new_position, self.maze.end)
            progress = prev_d - new_d  # positive if closer
            reward += self.reward_config.progress_scale * progress

        return reward

    def update_rewards(self, reward: int) -> None:
        """
        Update cumulative rewards and other statistics.
        
        :param reward: The reward to update.
        """
        self.cumulative_reward += reward
        if reward == self.evaluate_expression(self.reward_config.get_modifier('hit_wall')):
            self.times_hit_wall += 1
        elif reward == self.evaluate_expression(self.reward_config.get_modifier('revisit_non_optimal_path')):
            self.times_revisited_square += 1
        else:
            self.non_repeating_steps_taken += 1

    def reset_rewards(self) -> None:
        """
        Reset rewards and other statistics at the start of each episode.
        """
        self.cumulative_reward = 0
        self.times_hit_wall = 0
        self.times_revisited_square = 0
        self.non_repeating_steps_taken = 0
