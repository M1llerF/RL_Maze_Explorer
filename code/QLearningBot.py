import os
import pickle
import numpy as np
from typing import Any, Dict, Tuple

from BotStatistics import BotStatistics
from BaseBot import BaseBot
from BotTools import BotTools
from BotConfigs import QLearningConfig
from services.repository import ArtifactsRepository
from services.runners import QLearningEpisodeRunner


class QLearning:
    def __init__(self, q_learning_config: QLearningConfig, repo: ArtifactsRepository | None = None, profile_name: str | None = None):
        """Initialize Q-learning algorithm with the given configuration."""
        self.lr = q_learning_config.learning_rate
        self.gamma = q_learning_config.discount_factor
        self.num_actions = 4
        self.q_table: Dict[Any, np.ndarray] = {}
        self.initial_exploration_rate = 1.0
        self.min_exploration_rate = 0.1
        # Decay per step across episodes (persistent)
        self.exploration_decay_rate = 0.0005
        self.total_steps = 0
        self.use_position_in_state = getattr(q_learning_config, 'use_position_in_state', True)
        self._repo = repo
        self._profile = profile_name

    def update_q_value(self, state: Any, action: int, reward: float, new_state: Any) -> None:
        """ Update Q-value for the given state-action pair."""
        state_key = self.state_to_key(state)
        new_state_key = self.state_to_key(new_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        if new_state_key not in self.q_table:
            self.q_table[new_state_key] = np.zeros(self.num_actions)

        old_value = self.q_table[state_key][action]
        future_optimal_value = np.max(self.q_table[new_state_key])
        new_value = old_value + self.lr * (reward + self.gamma * future_optimal_value - old_value)
        self.q_table[state_key][action] = new_value
    
    def choose_action(self, state: Any, statistics: BotStatistics) -> int:
        """Choose action based on the exploration-exploitation trade-off."""
        state_key = self.state_to_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions) 
        
        # Use a persistent decay across all episodes to reduce exploration over time
        exploration_rate = max(
            self.min_exploration_rate,
            self.initial_exploration_rate - self.exploration_decay_rate * self.total_steps,
        )
        if np.random.rand() < exploration_rate:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state_key])
    
    def save_q_table(self) -> None:
        if self._repo and self._profile:
            try:
                self._repo.save_q_table(self._profile, self.q_table)
            except Exception:
                pass

    def load_q_table(self) -> None:
        if self._repo and self._profile:
            try:
                self.q_table = self._repo.load_q_table(self._profile) or {}
            except Exception:
                self.q_table = {}

    def state_to_key(self, state: Any) -> Tuple:
        """Convert the state to a hashable key for the Q-table."""
        position_index, wall_distances, goal_direction = state
        if self.use_position_in_state:
            return position_index, wall_distances, goal_direction
        # Exclude absolute position to improve generalization to new mazes
        return wall_distances, goal_direction

class QLearningBot(BaseBot):
    def __init__(self, maze, config, reward_system, statistics, profile_name, repository: ArtifactsRepository | None = None):
        """
        Initialize the Q-learning bot.

        :param maze: The maze object.
        :param config: Q-learning configuration.
        :param reward_system: Reward system for evaluating actions.
        :param statistics: Instance of BotStatistics for tracking statistics.
        :param profile_name: Name of the profile for saving/loading data.
        """
        super().__init__(maze, statistics, config)
        # Use injected repository (preferred), fallback to default for backward-compat
        self.repo = repository or ArtifactsRepository()
        self.q_learning = QLearning(config, repo=self.repo, profile_name=profile_name)
        self.tools = BotTools(maze)
        self.reward_system = reward_system
        self.profile_name = profile_name
        self.total_reward = 0
        self.episode_counter = 0
        self.position = maze.get_start()
        self.state = self.calculate_state()
        self.q_learning.load_q_table()  # Load Q-table when initializing

        try:
            self.repo.ensure_maze_file(profile_name)
            maze_data = self.repo.load_maze_data(profile_name)
        except Exception:
            maze_data = {"highest": {"reward": float('-inf')}, "lowest": {"reward": float('inf')}}
        self.highest_reward = float(maze_data.get("highest", {}).get("reward", float('-inf')))
        self.lowest_reward = float(maze_data.get("lowest", {}).get("reward", float('inf')))

        # Visualization step-wise execution state
        self._vis_active = False
        self._vis_initialized = False
        self._vis_optimal_path = None
        self._vis_optimal_length = 0
        self._vis_step_limit = 0
        self._vis_best_distance = 0
        self._vis_no_progress_steps = 0
        self._vis_progress_patience = 0
        self._vis_steps = 0
        self._vis_times_hit_wall = 0
        # Episode runner delegates training episode orchestration
        self.runner = QLearningEpisodeRunner(self)

    def get_bot_specific_data(self):
        """Retrieve bot-specific data."""
        return {'q_table': self.q_learning.q_table}
    
    def initialize_specific_data(self, data):
        """Initialize bot-specific data."""
        self.q_learning.q_table = data.get('q_table', {})
        self.q_learning.load_q_table()  # Load the Q-table from a file

    def calculate_state(self, position=None):
        """Calculate the state based on the given position (or current position)."""
        if position is None:
            position = self.position
        position_index = self.tools.pos_to_state(position)
        wall_distances, goal_direction = self.tools.detect_walls(position)
        # visited = self.statistics.get_visited_positions()
        # distance_to_goal = self.tools.get_distance_to_goal(self.position)
        # return (position_index, wall_distances, tuple(visited), distance_to_goal, goal_direction)
        return (position_index, wall_distances, goal_direction)
    
    def run_episode(self):
        """Run a single episode of Q-learning (delegated to episode runner)."""
        self.runner.run_episode()

    def reset(self):
        """Reset the bot's position, statistics, and Q-learning data."""
        self.position = self.maze.start
        self.statistics.reset()
        self.total_reward = 0
        self.state = self.calculate_state()

    # Training lifecycle hooks (explicit contract implementation)
    def on_episode_start(self, mode: str) -> None:
        if mode != "training":
            raise ValueError(f"Unsupported episode mode for QLearningBot: {mode}")

    def on_episode_step(self, mode: str, step_index: int) -> None:
        if mode != "training":
            raise ValueError(f"Unsupported episode mode for QLearningBot: {mode}")
        if step_index < 0:
            raise ValueError(f"step_index must be non-negative, got {step_index}")

    def on_episode_end(self, mode: str, outcome: str) -> None:
        if mode != "training":
            raise ValueError(f"Unsupported episode mode for QLearningBot: {mode}")
        if not outcome:
            raise ValueError("outcome must be a non-empty string")

    # ---- Visualization step-wise execution helpers ----
    def begin_visualization_episode(self):
        """Initialize state for a step-wise episode run used by visualization."""
        optimal_path = self.tools.get_optimal_path_info(self.maze.start, self.maze.end, output='path')
        optimal_length = len(optimal_path)
        area_bonus = int(0.5 * self.maze.width * self.maze.height)
        step_limit = min(5000, max(200, 12 * optimal_length + area_bonus)) if optimal_length > 0 else max(200, area_bonus)

        def manhattan(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        self.position = self.maze.get_start()
        self.statistics.reset()
        self.total_reward = 0
        self.state = self.calculate_state()

        self._vis_active = True
        self._vis_initialized = True
        self._vis_optimal_path = optimal_path
        self._vis_optimal_length = optimal_length
        self._vis_step_limit = step_limit
        self._vis_best_distance = manhattan(self.position, self.maze.end)
        self._vis_no_progress_steps = 0
        self._vis_progress_patience = min(200, 50 * optimal_length)
        self._vis_steps = 0
        self._vis_times_hit_wall = 0

    def step_visualization(self, max_steps: int = 1) -> bool:
        """
        Perform up to max_steps steps for live visualization only.

        Unified visualization semantics (singular policy across bots):
        - No learning/backprop or Q-table updates during visualization
        - No persistence of rewards, heatmaps, or profile counters
        - Only in-memory bot position and statistics.heatmap are updated for drawing
        - Returns True when the episode finishes; caller resets environment
        """
        if not self._vis_initialized:
            self.begin_visualization_episode()

        for _ in range(max_steps):
            if self.position == self.maze.end:
                return self._finalize_visualization_episode()

            action = self.q_learning.choose_action(self.state, self.statistics)
            new_position = self.tools.calculate_next_position(self.position, action)
            self.statistics.total_steps = self.statistics.times_revisited_squares + self.statistics.non_repeating_steps_taken

            reward = 0
            if not self.maze.is_valid_position(self.profile_name, new_position[0], new_position[1]):
                # In visualization: compute reward for display consistency, but do not learn
                reward += self.reward_system.get_reward(self.position, new_position, self._vis_optimal_path, self._vis_optimal_length, self.statistics.get_visited_positions())
                # Do not update Q-values or global training step counters here
                self.total_reward += reward
                self._vis_times_hit_wall += 1
                # continue to next step without moving
            else:
                # Track the previous position, then count the new cell after moving
                self.statistics.update_last_visited(self.position)
                reward += self.reward_system.get_reward(self.position, new_position, self._vis_optimal_path, self._vis_optimal_length, self.statistics.get_visited_positions())

                if new_position in self.statistics.get_visited_positions():
                    self.statistics.times_revisited_squares += 1
                else:
                    self.statistics.non_repeating_steps_taken += 1

                if self.statistics.total_steps > self._vis_step_limit:
                    reward += -100
                    self.total_reward += reward
                    return self._finalize_visualization_episode()

                self.total_reward += reward
                # Advance environment state only; do not learn
                new_state = self.calculate_state(new_position)
                self.position = new_position
                self.statistics.update_visited_positions(self.position)
                self.state = new_state
                self._vis_steps += 1

                # Progress tracking for early-stop (currently not used to break early)
                def manhattan(a, b):
                    return abs(a[0]-b[0]) + abs(a[1]-b[1])
                current_distance = manhattan(self.position, self.maze.end)
                if current_distance < self._vis_best_distance:
                    self._vis_best_distance = current_distance
                    self._vis_no_progress_steps = 0
                else:
                    self._vis_no_progress_steps += 1

            if self.position == self.maze.end:
                return self._finalize_visualization_episode()

        return False

    def _finalize_visualization_episode(self) -> bool:
        """
        Finalize a visualization-only episode.

        Unified policy: do NOT persist any training artifacts here. Leave
        Q-table, rewards log, and profile stats untouched. Only reset local
        visualization state so a fresh episode can start on next call.
        """
        self._vis_active = False
        self._vis_initialized = False
        return True
