from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VisualizationStrategy(ABC):
    @abstractmethod
    def visualize(self, frame: Any, snapshot: Any) -> None:
        pass


class DefaultVisualizationStrategy(VisualizationStrategy):
    """Render through a read-only visualization snapshot."""

    def visualize(self, frame: Any, snapshot: Any) -> None:
        frame.renderVisualizationSnapshot(snapshot)


# Legacy alias so any existing imports keep working.
QLearningBotVisualizationStrategy = DefaultVisualizationStrategy
