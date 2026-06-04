from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import ast
from environment.sensing import MazeSensingService
from environment.traversal import TraversalPolicy, build_traversal_policy


def clipRewardValue(reward: float, config: Any) -> float:
    """Optionally clamp a reward using config.rewardClipMin/rewardClipMax."""
    clipMin = getattr(config, "rewardClipMin", None)
    clipMax = getattr(config, "rewardClipMax", None)
    if clipMin is None or clipMax is None:
        return float(reward)
    return max(float(clipMin), min(float(clipMax), float(reward)))

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
            'new_tile_visited': '2',
            'move_in_optimal_path': '5',
            'see_goal_new_location': '50',
            'see_goal_revisit': '5',
            'per_move_penalty': '-1',
            'enemy_contact': '-250',
            'death_by_enemy': '-1000',
            'enemy_killed': '500',
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
    def __init__(
        self,
        maze: Any,
        *,
        traversal: TraversalPolicy | None = None,
        sensing_service: MazeSensingService | None = None,
    ) -> None:
        self.maze = maze
        self.traversal = traversal or build_traversal_policy(maze)
        self.sensing = sensing_service or MazeSensingService(maze=maze, traversal=self.traversal)

    def goalInSight(self, pos: tuple[int, int]) -> int:
        return self.sensing.goal_in_sight(pos)


@dataclass(frozen=True)
class RewardEnvironmentSnapshot:
    goal_position: tuple[int, int]
    traversal: TraversalPolicy
    sensors: MazeSensors


@dataclass(frozen=True)
class RewardStepContext:
    previous_position: tuple[int, int]
    new_position: tuple[int, int]
    optimal_path: list[tuple[int, int]]
    optimal_length: int
    visited_positions: dict[tuple[int, int], int]
    goal_position: tuple[int, int]
    hit_wall: bool
    goal_visible: bool
    semantic_events: tuple["RewardEvent", ...]


@dataclass(frozen=True)
class RewardEvent:
    name: str
    payload: dict[str, Any]


class RewardSystem:
    def __init__(
        self,
        maze: Any,
        rewardConfig: RewardConfig,
        sensors: MazeSensors | None = None,
        *,
        traversal: TraversalPolicy | None = None,
    ) -> None:
        self.maze = maze
        self.rewardConfig = rewardConfig
        self.cumulativeReward = 0
        self.timesHitWall = 0
        self.timesRevisitedSquare = 0
        self.nonRepeatingStepsTaken = 0
        self.traversal = traversal or build_traversal_policy(maze)
        self.sensors = sensors or MazeSensors(maze, traversal=self.traversal)
        self.environment = RewardEnvironmentSnapshot(
            goal_position=tuple(maze.end),
            traversal=self.traversal,
            sensors=self.sensors,
        )
    
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

    def buildStepContext(
        self,
        prevPosition: tuple[int, int],
        newPosition: tuple[int, int],
        optimalPath: list[tuple[int, int]],
        optimalLength: int,
        visitedPositions: dict[tuple[int, int], int],
        *,
        semanticEvents: tuple[RewardEvent, ...] = (),
    ) -> RewardStepContext:
        return RewardStepContext(
            previous_position=prevPosition,
            new_position=newPosition,
            optimal_path=optimalPath,
            optimal_length=optimalLength,
            visited_positions=visitedPositions,
            goal_position=self.environment.goal_position,
            hit_wall=not self.environment.traversal.is_valid_position(newPosition),
            goal_visible=bool(self.environment.sensors.goalInSight(newPosition)),
            semantic_events=tuple(semanticEvents),
        )

    def evaluateStep(self, step: RewardStepContext) -> float:
        reward = 0

        context = {
            'optimal_length': step.optimal_length,
            'visited_positions': step.visited_positions,
            'optimal_path': step.optimal_path,
            'new_position': step.new_position,
        }


        for key, expr in self.rewardConfig.rewardModifiers.items():
            # Use configured values directly (no scaling by path length)
            valueExpr = str(expr)

            if key == 'goal_reached' and step.new_position == step.goal_position:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'hit_wall' and step.hit_wall:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'revisit_optimal_path' and step.new_position in step.visited_positions and step.new_position in step.optimal_path:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'revisit_non_optimal_path' and step.new_position in step.visited_positions and step.new_position not in step.optimal_path:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'new_tile_visited' and not step.hit_wall and step.new_position not in step.visited_positions:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'move_in_optimal_path' and step.new_position in step.optimal_path:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'see_goal_new_location' and step.goal_visible and step.new_position not in step.visited_positions:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'see_goal_revisit' and step.goal_visible and step.new_position in step.visited_positions:
                reward += self.evaluateExpression(valueExpr, **context)
            elif key == 'per_move_penalty':
                reward += self.evaluateExpression(valueExpr, **context)
        
        # Potential-based shaping: reward progress toward goal (distance reduction)
        if self.rewardConfig.usePotentialShaping:
            # Manhattan distance tends to be stable in grid mazes
            def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
                return abs(a[0]-b[0]) + abs(a[1]-b[1])
            prevD = manhattan(step.previous_position, step.goal_position)
            newD = manhattan(step.new_position, step.goal_position)
            progress = prevD - newD  # positive if closer
            reward += self.rewardConfig.progressScale * progress

        reward += self.evaluateSemanticEvents(step.semantic_events)
        return reward

    def evaluateSemanticEvents(self, events: tuple["RewardEvent", ...]) -> float:
        reward = 0.0
        semanticEventCounts = self._semanticEventCounts(events)
        context = {'semantic_event_counts': semanticEventCounts}
        for key, count in semanticEventCounts.items():
            expr = self.rewardConfig.rewardModifiers.get(key)
            if expr is None:
                continue
            reward += count * self.evaluateExpression(str(expr), **context)
        return reward

    @staticmethod
    def _semanticEventCounts(events: tuple[RewardEvent, ...]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in events:
            name = str(event.name)
            counts[name] = counts.get(name, 0) + 1
        return counts

    def getReward(
        self,
        prevPosition: tuple[int, int],
        newPosition: tuple[int, int],
        optimalPath: list[tuple[int, int]],
        optimalLength: int,
        visitedPositions: dict[tuple[int, int], int],
    ) -> float:
        """
        Compatibility shim for legacy callers. New code should build a RewardStepContext
        and call evaluateStep().
        """
        return self.evaluateStep(
            self.buildStepContext(
                prevPosition,
                newPosition,
                optimalPath,
                optimalLength,
                visitedPositions,
            )
        )

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
