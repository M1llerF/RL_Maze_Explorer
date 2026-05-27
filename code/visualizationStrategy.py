from abc import ABC, abstractmethod
from typing import Any

class VisualizationStrategy(ABC):
    @abstractmethod
    def visualize(self, frame: Any, bot: Any, profileIndex: int) -> None:
        pass

class QLearningBotVisualizationStrategy(VisualizationStrategy):
    def visualize(self, frame: Any, bot: Any, profileIndex: int) -> None:
        selectedProfile = frame.profileSelect.get()
        # Load via centralized repository on the bot
        try:
            repo = getattr(bot, 'repo', None)
            mazeData: dict[str, Any] = repo.loadMazeData(selectedProfile) if repo else {}
        except Exception:
            mazeData = {}

        # Check if maze_data contains the required keys
        if "latest" in mazeData and "highest" in mazeData and "lowest" in mazeData:
            # Update bot's maze_data
            bot.mazeData = mazeData
            
            # Display heatmaps
            frame.displayHeatmap(frame.heatmapCanvasLatest, mazeData["latest"].get("maze"), mazeData["latest"].get("start"), mazeData["latest"].get("end"), mazeData["latest"].get("heatmap_data"))
            frame.displayHeatmap(frame.heatmapCanvasHighest, mazeData["highest"].get("maze"), mazeData["highest"].get("start"), mazeData["highest"].get("end"), mazeData["highest"].get("heatmap_data"))
            frame.displayHeatmap(frame.heatmapCanvasLowest, mazeData["lowest"].get("maze"), mazeData["lowest"].get("start"), mazeData["lowest"].get("end"), mazeData["lowest"].get("heatmap_data"))
        else:
            # Gracefully skip if data is missing; UI will remain unchanged
            return

        frame.displayQtable(bot, profileIndex)
        frame.displayStatistics(bot, profileIndex)
        frame.displayRewardGraph(bot)
