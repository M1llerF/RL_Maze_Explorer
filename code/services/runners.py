from __future__ import annotations

from typing import Any


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
        tools = bot.tools
        maze = bot.maze
        stats = bot.statistics
        rsys = bot.reward_system
        ql = bot.q_learning

        outcome = "aborted"
        bot.on_episode_start("training")
        try:
            optimal_path = tools.get_optimal_path_info(maze.start, maze.end, output='path')
            optimal_length = len(optimal_path)
            area_bonus = int(0.5 * maze.width * maze.height)
            step_limit = (min(5000, max(200, 12 * optimal_length + area_bonus)) if optimal_length > 0 else max(200, area_bonus))

            def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
                return abs(a[0] - b[0]) + abs(a[1] - b[1])

            bot.total_reward = 0
            best_distance = manhattan(bot.position, maze.end)
            steps = 0
            times_hit_wall = 0

            while bot.position != maze.end:
                bot.on_episode_step("training", steps)
                reward = 0
                action = ql.choose_action(bot.state, stats)
                new_position = tools.calculate_next_position(bot.position, action)
                stats.total_steps = stats.times_revisited_squares + stats.non_repeating_steps_taken

                if not maze.is_valid_position(bot.profile_name, new_position[0], new_position[1]):
                    reward += rsys.get_reward(bot.position, new_position, optimal_path, optimal_length, stats.get_visited_positions())
                    new_state = bot.calculate_state()
                    ql.update_q_value(bot.state, action, reward, new_state)
                    bot.total_reward += reward
                    times_hit_wall += 1
                    ql.total_steps += 1
                    continue

                stats.update_last_visited(bot.position)
                reward += rsys.get_reward(bot.position, new_position, optimal_path, optimal_length, stats.get_visited_positions())

                if new_position in stats.get_visited_positions():
                    stats.times_revisited_squares += 1
                else:
                    stats.non_repeating_steps_taken += 1

                if stats.total_steps > step_limit:
                    reward += -100
                    new_state = bot.calculate_state()
                    ql.update_q_value(bot.state, action, reward, new_state)
                    bot.total_reward += reward
                    outcome = "step_limit_statistics"
                    break

                bot.total_reward += reward
                new_state = bot.calculate_state(new_position)
                ql.update_q_value(bot.state, action, reward, new_state)
                ql.total_steps += 1

                bot.position = new_position
                stats.update_visited_positions(bot.position)
                bot.state = new_state
                steps += 1

                current_distance = manhattan(bot.position, maze.end)
                if current_distance < best_distance:
                    best_distance = current_distance
                if steps > step_limit:
                    outcome = "step_limit_loop"
                    break

            if bot.position == maze.end:
                outcome = "goal_reached"

            heatmap_data = stats.get_visited_positions()
            try:
                bot.repo.save_maze_episode(bot.profile_name, maze, heatmap_data, bot.total_reward)
                bot.repo.update_steps_from_heatmap(bot.profile_name, heatmap_data)
                if times_hit_wall:
                    bot.repo.increment_times_hit_wall(bot.profile_name, times_hit_wall)
            except Exception:
                pass
            bot.repo.append_reward(bot.profile_name, bot.total_reward)

            bot.episode_counter += 1
            ql.save_q_table()
        finally:
            bot.on_episode_end("training", outcome)

