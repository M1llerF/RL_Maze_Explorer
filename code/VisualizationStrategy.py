from abc import ABC, abstractmethod
from typing import Any

class VisualizationStrategy(ABC):
    @abstractmethod
    def visualize(self, frame: Any, bot: Any, profile_index: int):
        pass

class QLearningBotVisualizationStrategy(VisualizationStrategy):
    def visualize(self, frame, bot, profile_index):
        selected_profile = frame.profile_select.get()
        # Load via centralized repository on the bot
        try:
            repo = getattr(bot, 'repo', None)
            maze_data = repo.load_maze_data(selected_profile) if repo else {}
        except Exception:
            maze_data = {}

        # Check if maze_data contains the required keys
        if "latest" in maze_data and "highest" in maze_data and "lowest" in maze_data:
            # Update bot's maze_data
            bot.maze_data = maze_data
            
            # Display heatmaps
            frame.display_heatmap(frame.heatmap_canvas_latest, maze_data["latest"].get("maze"), maze_data["latest"].get("start"), maze_data["latest"].get("end"), maze_data["latest"].get("heatmap_data"))
            frame.display_heatmap(frame.heatmap_canvas_highest, maze_data["highest"].get("maze"), maze_data["highest"].get("start"), maze_data["highest"].get("end"), maze_data["highest"].get("heatmap_data"))
            frame.display_heatmap(frame.heatmap_canvas_lowest, maze_data["lowest"].get("maze"), maze_data["lowest"].get("start"), maze_data["lowest"].get("end"), maze_data["lowest"].get("heatmap_data"))
        else:
            # Gracefully skip if data is missing; UI will remain unchanged
            return

        frame.display_qtable(bot, profile_index)
        frame.display_statistics(bot, profile_index)
        frame.display_reward_graph(bot)
