# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np

from DisplayTools import DisplayTools
from RewardGrapher import RewardGrapher
from VisualizationStrategy import QLearningBotVisualizationStrategy


class VisualizationWindow(tk.Toplevel):
    def __init__(self, parent: Any, game_env: Any, profile_name: str, profile_index: int) -> None:
        super().__init__(parent)
        self.game_env = game_env
        self.profile_name = profile_name
        self.profile_index = profile_index
        self.title("Maze Visualization")
        self.geometry("600x600")

        banner = tk.Label(self, text="Training is PAUSED for this profile while this window is open.",
                          bg="#fff3cd", fg="#856404", anchor="center")
        banner.pack(fill=tk.X)

        self.canvas = tk.Canvas(self, width=500, height=500, bg="white")
        self.canvas.pack(pady=20)

        controls = tk.Frame(self)
        controls.pack(pady=5)
        tk.Label(controls, text="Steps per frame:").pack(side=tk.LEFT)
        self.steps_var = tk.IntVar(value=20)
        self.steps_scale = tk.Scale(controls, from_=1, to=200, orient=tk.HORIZONTAL, variable=self.steps_var, length=200)
        self.steps_scale.pack(side=tk.LEFT, padx=5)
        self.paused = False
        self.pause_btn = ttk.Button(controls, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.after_id = None
        self.visualize = True
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        try:
            self.game_env.pause_training_for(self.profile_name)
        except Exception:
            pass

        self.update_visualization()

    def update_visualization(self) -> None:
        if not self.visualize:
            return
        self.canvas.delete("all")
        bot = self.game_env.bots[self.profile_index]
        if not self.paused:
            try:
                finished = bot.step_visualization(max_steps=self.steps_var.get())
                if finished:
                    self.game_env.reset_environment(self.profile_index)
            except AttributeError:
                pass
        bot_position = bot.position
        self.display_with_bot_and_heatmap(bot_position, bot.statistics.get_visited_positions())
        self.after_id = self.after(100, self.update_visualization)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_btn.configure(text="Resume" if self.paused else "Pause")

    def display_with_bot(self, bot_position: tuple[int, int]) -> None:
        maze = self.game_env.maze
        cell_width = self.canvas.winfo_width() / maze.width
        cell_height = self.canvas.winfo_height() / maze.height
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.grid[y][x] == 1:
                    self.canvas.create_rectangle(x * cell_width, y * cell_height,
                                                 (x + 1) * cell_width, (y + 1) * cell_height,
                                                 fill="black")
        start = maze.get_start()
        end = maze.end
        self.canvas.create_rectangle(start[1] * cell_width, start[0] * cell_height,
                                     (start[1] + 1) * cell_width, (start[0] + 1) * cell_height,
                                     fill="blue")
        self.canvas.create_rectangle(end[1] * cell_width, end[0] * cell_height,
                                     (end[1] + 1) * cell_width, (end[0] + 1) * cell_height,
                                     fill="green")
        self.canvas.create_oval(bot_position[1] * cell_width, bot_position[0] * cell_height,
                                (bot_position[1] + 1) * cell_width, (bot_position[0] + 1) * cell_height,
                                fill="red")

    def display_with_bot_and_heatmap(self, bot_position: tuple[int, int], visited_positions: dict[tuple[int, int], int]) -> None:
        maze = self.game_env.maze
        cell_width = self.canvas.winfo_width() / maze.width
        cell_height = self.canvas.winfo_height() / maze.height
        heatmap = np.zeros((maze.height, maze.width))
        for (x, y), count in visited_positions.items():
            heatmap[x, y] = count
        max_heat = heatmap.max() if heatmap.max() > 0 else 1
        cmap = plt.get_cmap("Reds")
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.grid[y][x] == 1:
                    self.canvas.create_rectangle(x * cell_width, y * cell_height,
                                                 (x + 1) * cell_width, (y + 1) * cell_height,
                                                 fill="black")
                else:
                    heat_value = heatmap[y, x] / max_heat
                    if heat_value > 0:
                        color = mcolors.to_hex(cmap(heat_value))
                        self.canvas.create_rectangle(x * cell_width, y * cell_height,
                                                     (x + 1) * cell_width, (y + 1) * cell_height,
                                                     fill=color, outline=color)
        start = maze.get_start()
        end = maze.end
        self.canvas.create_rectangle(start[1] * cell_width, start[0] * cell_height,
                                     (start[1] + 1) * cell_width, (start[0] + 1) * cell_height,
                                     fill="blue")
        self.canvas.create_rectangle(end[1] * cell_width, end[0] * cell_height,
                                     (end[1] + 1) * cell_width, (end[0] + 1) * cell_height,
                                     fill="green")
        self.canvas.create_oval(bot_position[1] * cell_width, bot_position[0] * cell_height,
                                (bot_position[1] + 1) * cell_width, (bot_position[0] + 1) * cell_height,
                                fill="red")

    def on_close(self) -> None:
        self.visualize = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        try:
            self.game_env.resume_training_for(self.profile_name)
        except Exception:
            pass
        try:
            controller = getattr(self.master, 'controller', None)
            if controller and hasattr(controller, 'frames') and 'BotTrainingFrame' in controller.frames:
                controller.frames['BotTrainingFrame'].status_hint.configure(text="")
        except Exception:
            pass
        self.destroy()


class VisualizationFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller
        self.visualization_strategies = {
            'QLearningBot': QLearningBotVisualizationStrategy(),
        }
        self.canvas_agg: Any = None

        ttk.Label(self, text="Visualizations", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)
        self.canvas = tk.Canvas(self, height=600, width=1000)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        def _wheel_scroll(event: Any) -> str:
            units = -1 if getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4 else 1
            try:
                self.canvas.yview_scroll(units, 'units')
            except Exception:
                pass
            return "break"
        self.canvas.bind('<MouseWheel>', _wheel_scroll)
        self.canvas.bind('<Button-4>', _wheel_scroll)
        self.canvas.bind('<Button-5>', _wheel_scroll)

        ttk.Label(self.scrollable_frame, text="Select Profile:").pack()
        self.profile_select = ttk.Combobox(self.scrollable_frame)
        self.profile_select.pack()

        ttk.Button(self.scrollable_frame, text="Load Profile", command=self.load_profile).pack(pady=10)
        self.heatmap_frame = tk.Frame(self.scrollable_frame)
        self.heatmap_frame.pack(pady=10)
        ttk.Label(self.heatmap_frame, text="Latest Maze").grid(row=0, column=0, pady=10)
        self.heatmap_canvas_latest = tk.Canvas(self.heatmap_frame, width=300, height=300, bg="white")
        self.heatmap_canvas_latest.grid(row=1, column=0, padx=5)
        ttk.Label(self.heatmap_frame, text="Highest Reward Maze").grid(row=0, column=1, pady=10)
        self.heatmap_canvas_highest = tk.Canvas(self.heatmap_frame, width=300, height=300, bg="white")
        self.heatmap_canvas_highest.grid(row=1, column=1, padx=5)
        ttk.Label(self.heatmap_frame, text="Lowest Reward Maze").grid(row=0, column=2, pady=10)
        self.heatmap_canvas_lowest = tk.Canvas(self.heatmap_frame, width=300, height=300, bg="white")
        self.heatmap_canvas_lowest.grid(row=1, column=2, padx=5)
        ttk.Label(self.scrollable_frame, text="Reward Graph Visualization:").pack(pady=10)
        self.reward_canvas = tk.Canvas(self.scrollable_frame, width=800, height=400)
        self.reward_canvas.pack(pady=10)
        ttk.Label(self.scrollable_frame, text="Q-Table Visualization:").pack(pady=10)
        self.qtable_output = tk.Text(self.scrollable_frame, height=10, width=50)
        self.qtable_output.pack(pady=10)
        self.qtable_scrollbar = ttk.Scrollbar(self.scrollable_frame, command=self.qtable_output.yview)
        self.qtable_scrollbar.pack(side="right", fill="y")
        self.qtable_output.config(yscrollcommand=self.qtable_scrollbar.set)
        ttk.Label(self.scrollable_frame, text="Statistics:").pack(pady=10)
        self.statistics_output = tk.Text(self.scrollable_frame, height=5, width=50)
        self.statistics_output.pack(pady=10)
        self.load_profiles()

    def on_frame_configure(self, event: Any) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def load_profiles(self) -> None:
        profiles = self.controller.game_env.profile_manager.list_profiles()
        self.profile_select['values'] = profiles

    def load_profile(self) -> None:
        selected_profile = self.profile_select.get()
        if not selected_profile:
            messagebox.showerror("Error", "No profile selected.")
            return
        profile = self.controller.game_env.profile_manager.load_profile(selected_profile)
        profile_index = self.controller.game_env.apply_profile(profile)
        bot = self.controller.game_env.bots[profile_index]
        strategy = self.visualization_strategies.get(profile.bot_type)
        if strategy:
            strategy.visualize(self, bot, profile_index)

    def display_heatmap(self, canvas: Any, maze: Any, start: Any, end: Any, heatmap_data: Any) -> None:
        try:
            DisplayTools.display_heatmap(canvas, maze, start, end, heatmap_data)
        except Exception:
            pass

    def display_qtable(self, bot: Any, profile_index: int) -> None:
        self.qtable_output.delete("1.0", tk.END)
        if hasattr(bot, 'q_learning') and hasattr(bot.q_learning, 'q_table'):
            top_values = self.get_top_q_values(bot, profile_index)
            self.qtable_output.insert(tk.END, "Top Q-Table Values:\n")
            for i, (q_value, (state, actions)) in enumerate(top_values):
                position, surrounding, step_count = state
                best_action_index = int(np.argmax(actions))
                best_action = self.get_action_label(best_action_index)
                best_q_value = q_value
                self.qtable_output.insert(tk.END, f"Rank {i+1}:\n")
                self.qtable_output.insert(tk.END, f"  Current Position: {position}\n")
                self.qtable_output.insert(tk.END, f"  Surrounding: {surrounding}\n")
                self.qtable_output.insert(tk.END, f"  Step Count: {step_count}\n")
                self.qtable_output.insert(tk.END, f"  Best Action: {best_action}\n")
                self.qtable_output.insert(tk.END, f"  Best Q-value: {best_q_value}\n\n")
        else:
            self.qtable_output.insert(tk.END, "No tabular Q-table available for this bot.\n")

    def display_statistics(self, bot: Any, profile_index: int) -> None:
        self.statistics_output.delete("1.0", tk.END)
        try:
            repo = self.controller.game_env.repository
            profile_data = repo.read_profile_stats(bot.profile_name) or {}
        except Exception:
            profile_data = {}
        self.statistics_output.insert(tk.END, f"Total Steps: {profile_data.get('total_steps', 0)}\n")
        self.statistics_output.insert(tk.END, f"Non-Repeating Steps: {profile_data.get('non_repeating_steps_taken', 0)}\n")
        self.statistics_output.insert(tk.END, f"Times Revisited Squares: {profile_data.get('times_revisited_squares', 0)}\n")
        self.statistics_output.insert(tk.END, f"Times Bot Hit Wall: {profile_data.get('times_hit_wall', 0)}\n")

    def display_reward_graph(self, bot: Any) -> None:
        if self.canvas_agg:
            self.canvas_agg.get_tk_widget().destroy()
        try:
            reward_path = self.controller.game_env.repository.rewards_path(bot.profile_name)
        except Exception:
            reward_path = f'profiles/{bot.profile_name}/SimulationRewards.txt'
        reward_filenames = [reward_path]
        grapher = RewardGrapher(reward_filenames)
        self.canvas_agg = grapher.run(self.reward_canvas)

    def get_top_q_values(self, bot: Any, profile_index: int, n: int = 10) -> list[Any]:
        q_table = bot.q_learning.q_table
        q_table_items = list(q_table.items())
        top_items = []
        for item in q_table_items:
            q_value = np.max(item[1])
            if len(top_items) < n:
                top_items.append((q_value, item))
                top_items.sort(reverse=True, key=lambda item: item[0])
            else:
                if q_value > top_items[-1][0]:
                    top_items[-1] = (q_value, item)
                    top_items.sort(reverse=True, key=lambda item: item[0])
        return top_items

    def get_action_label(self, action_index: int) -> str:
        action_labels = ["Up", "Down", "Left", "Right"]
        return action_labels[action_index]
