from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeRuntime:
    optimal_path: list[tuple[int, int]]
    optimal_length: int
    step_limit: int
    steps: int = 0
    times_hit_wall: int = 0
    outcome: str = "aborted"
    best_distance: int = 0


class QLearningEpisodeRunner:
    """
    Episode controller for QLearningBot (training episodes only).

    Contract:
    - Owns episode orchestration (start/step/end loop control).
    - Delegates algorithm internals to the bot (policy/value updates).
    - Triggers persistence through repository calls at episode end.
    - Must not be used for visualization; visualization remains inference-only.
    """

    def __init__(self, bot: Any):
        self.bot = bot

    def run_episode(self) -> None:
        bot = self.bot
        bot.on_episode_start("training")
        runtime = self._start_episode()
        try:
            while bot.position != bot.maze.end:
                if not self._run_step(runtime):
                    break
            if bot.position == bot.maze.end:
                runtime.outcome = "goal_reached"
            self._finalize_episode(runtime)
        finally:
            bot.on_episode_end("training", runtime.outcome)

    def _start_episode(self) -> EpisodeRuntime:
        bot = self.bot
        tools = bot.tools
        maze = bot.maze

        optimal_path = tools.get_optimal_path_info(maze.start, maze.end, output='path')
        optimal_length = len(optimal_path)
        area_bonus = int(0.5 * maze.width * maze.height)
        step_limit = min(5000, max(200, 12 * optimal_length + area_bonus)) if optimal_length > 0 else max(200, area_bonus)

        bot.total_reward = 0
        return EpisodeRuntime(
            optimal_path=optimal_path,
            optimal_length=optimal_length,
            step_limit=step_limit,
            best_distance=self._manhattan(bot.position, maze.end),
        )

    def _run_step(self, runtime: EpisodeRuntime) -> bool:
        bot = self.bot
        tools = bot.tools
        maze = bot.maze
        stats = bot.statistics
        rsys = bot.reward_system
        ql = bot.q_learning

        bot.on_episode_step("training", runtime.steps)
        reward = 0
        action = ql.choose_action(bot.state, stats)
        new_position = tools.calculate_next_position(bot.position, action)
        stats.total_steps = stats.times_revisited_squares + stats.non_repeating_steps_taken

        if not maze.is_valid_position(bot.profile_name, new_position[0], new_position[1]):
            reward += rsys.get_reward(bot.position, new_position, runtime.optimal_path, runtime.optimal_length, stats.get_visited_positions())
            new_state = bot.calculate_state()
            ql.update_q_value(bot.state, action, reward, new_state)
            bot.total_reward += reward
            runtime.times_hit_wall += 1
            ql.total_steps += 1
            return True

        stats.update_last_visited(bot.position)
        reward += rsys.get_reward(bot.position, new_position, runtime.optimal_path, runtime.optimal_length, stats.get_visited_positions())

        if new_position in stats.get_visited_positions():
            stats.times_revisited_squares += 1
        else:
            stats.non_repeating_steps_taken += 1

        if stats.total_steps > runtime.step_limit:
            reward += -100
            new_state = bot.calculate_state()
            ql.update_q_value(bot.state, action, reward, new_state)
            bot.total_reward += reward
            runtime.outcome = "step_limit_statistics"
            return False

        bot.total_reward += reward
        new_state = bot.calculate_state(new_position)
        ql.update_q_value(bot.state, action, reward, new_state)
        ql.total_steps += 1

        bot.position = new_position
        stats.update_visited_positions(bot.position)
        bot.state = new_state
        runtime.steps += 1

        current_distance = self._manhattan(bot.position, maze.end)
        if current_distance < runtime.best_distance:
            runtime.best_distance = current_distance
        if runtime.steps > runtime.step_limit:
            runtime.outcome = "step_limit_loop"
            return False
        return True

    def _finalize_episode(self, runtime: EpisodeRuntime) -> None:
        bot = self.bot
        maze = bot.maze
        heatmap_data = bot.statistics.get_visited_positions()
        try:
            bot.repo.save_maze_episode(bot.profile_name, maze, heatmap_data, bot.total_reward)
            bot.repo.update_steps_from_heatmap(bot.profile_name, heatmap_data)
            if runtime.times_hit_wall:
                bot.repo.increment_times_hit_wall(bot.profile_name, runtime.times_hit_wall)
        except Exception:
            pass
        bot.repo.append_reward(bot.profile_name, bot.total_reward)
        bot.episode_counter += 1
        bot.q_learning.save_q_table()

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

