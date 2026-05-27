from typing import Dict, Tuple


class BotStatistics:
    """In-memory episode statistics only.

    Persistence and aggregation are handled by services.repository.ArtifactsRepository.
    """
    def __init__(self) -> None:
        self.totalSteps: int = 0
        self.timesHitWall: int = 0
        self.timesRevisitedSquares: int = 0
        self.nonRepeatingStepsTaken: int = 0
        self.visitedPositions: Dict[Tuple[int, int], int] = {}
        self.lastVisitedPositions: list[Tuple[int, int]] = []

    def reset(self) -> None:
        self.totalSteps = 0
        self.timesHitWall = 0
        self.timesRevisitedSquares = 0
        self.nonRepeatingStepsTaken = 0
        self.visitedPositions.clear()
        self.lastVisitedPositions.clear()

    def updateVisitedPositions(self, position: Tuple[int, int]) -> None:
        """Update the count of times a position has been visited."""
        self.visitedPositions[position] = self.visitedPositions.get(position, 0) + 1

    def getVisitedPositions(self) -> Dict[Tuple[int, int], int]:
        """Retrieve the dictionary of visited positions."""
        return self.visitedPositions

    def updateLastVisited(self, position: Tuple[int, int]) -> None:
        """Update the list of the last visited positions."""
        if len(self.lastVisitedPositions) >= 5:
            self.lastVisitedPositions.pop(0)
        if position not in self.lastVisitedPositions:
            self.lastVisitedPositions.append(position)

    def getLastVisited(self) -> list[Tuple[int, int]]:
        """Retrieve the list of the last visited positions."""
        return self.lastVisitedPositions
