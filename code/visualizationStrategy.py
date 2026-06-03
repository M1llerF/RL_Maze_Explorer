from abc import ABC, abstractmethod
from typing import Any


class VisualizationStrategy(ABC):
    @abstractmethod
    def visualize(self, frame: Any, bot: Any, profileIndex: int) -> None:
        pass


class DefaultVisualizationStrategy(VisualizationStrategy):
    """Universal strategy — works for any bot type that persists maze and reward data."""

    def visualize(self, frame: Any, bot: Any, profileIndex: int) -> None:
        selectedProfile = frame.profileSelect.get()
        try:
            repo = getattr(bot, 'repo', None)
            mazeData: dict[str, Any] = repo.loadMazeData(selectedProfile) if repo else {}
        except Exception:
            mazeData = {}

        if "latest" in mazeData and "highest" in mazeData and "lowest" in mazeData:
            bot.mazeData = mazeData
            frame.displayHeatmap(frame.heatmapCanvasLatest, mazeData["latest"].get("maze"), mazeData["latest"].get("start"), mazeData["latest"].get("end"), mazeData["latest"].get("heatmap_data"))
            frame.displayHeatmap(frame.heatmapCanvasHighest, mazeData["highest"].get("maze"), mazeData["highest"].get("start"), mazeData["highest"].get("end"), mazeData["highest"].get("heatmap_data"))
            frame.displayHeatmap(frame.heatmapCanvasLowest, mazeData["lowest"].get("maze"), mazeData["lowest"].get("start"), mazeData["lowest"].get("end"), mazeData["lowest"].get("heatmap_data"))

        frame.displayQtable(bot, profileIndex)
        frame.displayStatistics(bot, profileIndex)
        frame.displayRewardGraph(bot)


# Legacy alias so any existing imports keep working.
QLearningBotVisualizationStrategy = DefaultVisualizationStrategy
