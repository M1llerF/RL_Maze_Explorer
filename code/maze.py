import numpy as np
import random
from typing import Any
from environment.entities import entity_from_state, entity_to_state
from pathfinding import Pathfinding


MazeState = dict[str, Any]

class Maze:
    def __init__(self, width: int, height: int, start: tuple[int, int] | None = None, end: tuple[int, int] | None = None):
        """
        Initialize the maze with given dimensions and optionally set start and end points.
        """
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.start = start
        self.end = end
        self.entities: list[dict[str, Any]] = []
        self.minimumDistance = max(width, height) // 2
        self.setupSimpleMaze()
        # Do not create a matplotlib figure here; the app uses Tk canvas.
        # Leaving interactive figure creation disabled prevents stray empty windows.
    
    def isValidPosition(self, botName: str | None, x: int, y: int) -> bool:
        """
        Check if a position is valid (within bounds and not a wall).
        Pure check: no side effects.
        """
        return 0 <= x < self.height and 0 <= y < self.width and self.grid[x][y] == 0

    def setWall(self, x: int, y: int) -> None:
        """
        Set a wall at the specified position.
        Raise a error if the position is out of bounds.
        """
        if 0 <= x < self.height and 0 <= y < self.width:
            self.grid[x][y] = 1
        else:
            raise ValueError("Position out of maze bounds")
    
    def getStart(self) -> tuple[int, int]:
        """
        Get the start position of the maze.
        """
        if self.start is None:
            raise ValueError("Maze start is not set")
        return self.start
    
    def setStart(self, x: int, y: int) -> None:
        if self.isValidPosition(None, x, y):
            self.start = (x, y)
        else:
            raise ValueError("Invalid start position")
    
    def setGoal(self, x: int, y: int) -> None:
        """
        Set the goal position of the maze if it's valid.
        Raise a error if the position is invalid.
        """
        if self.isValidPosition(None, x, y):
            self.end = (x, y)
        else:
            raise ValueError("Invalid goal position")
        
    def setupSimpleMaze(
        self,
        minGenerationLength: int | None = None,
        maxGenerationLength: int | None = None,
    ) -> None:
        minLength = max(0, int(minGenerationLength)) if minGenerationLength is not None else None
        maxLength = max(0, int(maxGenerationLength)) if maxGenerationLength is not None else None
        if minLength is not None and maxLength is not None and minLength > maxLength:
            raise ValueError("minGenerationLength must be <= maxGenerationLength")

        fallbackState: MazeState | None = None
        fallbackDistance: int | None = None

        for _ in range(100):
            self._generateSimpleMazeTopology()
            pathLength = len(Pathfinding.aStarSearch(self, self.start, self.end))
            if self._matchesGenerationLength(pathLength, minLength, maxLength):
                return
            distance = self._generationLengthDistance(pathLength, minLength, maxLength)
            if fallbackDistance is None or distance < fallbackDistance:
                fallbackDistance = distance
                fallbackState = self.getState()

        if fallbackState is not None:
            self.setState(fallbackState)

    def _generateSimpleMazeTopology(self) -> None:
        # Initialize grid with walls
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
        self.entities = []

        def dfsIterative(x: int, y: int) -> None:
            stack = [(x, y)]
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

            while stack:
                cx, cy = stack[-1]
                neighbors: list[tuple[int, int, int, int]] = []
                for dx, dy in directions:
                    nx, ny = cx + 2 * dx, cy + 2 * dy
                    if 1 <= nx < self.height - 1 and 1 <= ny < self.width - 1 and self.grid[nx][ny] == 1:
                        neighbors.append((dx, dy, nx, ny))

                if not neighbors:
                    stack.pop()
                    continue

                dx, dy, nx, ny = random.choice(neighbors)
                self.grid[cx + dx][cy + dy] = 0
                self.grid[nx][ny] = 0
                stack.append((nx, ny))

        startX = random.randrange(1, self.height - 1, 2)
        startY = random.randrange(1, self.width - 1, 2)
        self.grid[startX][startY] = 0

        dfsIterative(startX, startY)

        startPositions = [(x, y) for x in range(1, self.height - 1) for y in range(1, self.width - 1) if self.grid[x][y] == 0]
        self.start = random.choice(startPositions)
        self.end = self.getFarthestValidEndPosition(startPositions)
        self.setStart(*self.start)
        self.setGoal(*self.end)

    @staticmethod
    def _matchesGenerationLength(
        pathLength: int,
        minLength: int | None,
        maxLength: int | None,
    ) -> bool:
        if minLength is not None and pathLength < minLength:
            return False
        if maxLength is not None and pathLength > maxLength:
            return False
        return True

    @staticmethod
    def _generationLengthDistance(
        pathLength: int,
        minLength: int | None,
        maxLength: int | None,
    ) -> int:
        if minLength is not None and pathLength < minLength:
            return int(minLength - pathLength)
        if maxLength is not None and pathLength > maxLength:
            return int(pathLength - maxLength)
        return 0

    def resize(self, width: int, height: int, regenerate: bool = True) -> None:
        """Resize the maze dimensions and optionally regenerate topology."""
        newWidth = max(5, int(width))
        newHeight = max(5, int(height))
        self.width = newWidth
        self.height = newHeight
        self.minimumDistance = max(newWidth, newHeight) // 2
        if regenerate:
            self.setupSimpleMaze()

    def setEntities(self, entities: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for raw in list(entities or []):
            if not isinstance(raw, dict):
                continue
            entity = entity_from_state(raw)
            row, col = entity.position
            if not self.isValidPosition(None, row, col):
                raise ValueError(f"Invalid entity position for maze topology: {(row, col)}")
            if self.start is not None and (row, col) == tuple(self.start):
                raise ValueError("Entity cannot occupy the maze start position")
            if self.end is not None and (row, col) == tuple(self.end):
                raise ValueError("Entity cannot occupy the maze goal position")
            normalized.append(entity_to_state(entity))
        self.entities = normalized

    def addEntity(self, entityType: str, position: tuple[int, int], **kwargs: Any) -> None:
        nextState = {"type": str(entityType), "position": [int(position[0]), int(position[1])]}
        nextState.update(kwargs)
        self.setEntities([*self.entities, nextState])

    def getState(self) -> MazeState:
        """Return a serializable snapshot of the current maze state."""
        return {
            'width': self.width,
            'height': self.height,
            'grid': [row[:] for row in self.grid],
            'start': tuple(self.start) if self.start is not None else None,
            'end': tuple(self.end) if self.end is not None else None,
            'entities': [dict(entity) for entity in self.entities],
        }

    def setState(self, state: MazeState) -> None:
        """Load a snapshot produced by get_state()."""
        self.width = int(state['width'])
        self.height = int(state['height'])
        self.grid = [row[:] for row in state['grid']]
        self.start = tuple(state['start']) if state['start'] is not None else None
        self.end = tuple(state['end']) if state['end'] is not None else None
        self.entities = []
        self.setEntities(list(state.get('entities', [])))

    def getFarthestValidEndPosition(self, startPositions: list[tuple[int, int]]) -> tuple[int, int]:
        """
        Get the farthest valid end position that is at least minimum_distance away from the start.
        """
        validEndPositions = [
            pos for pos in startPositions if np.linalg.norm(np.array(pos) - np.array(self.start)) >= self.minimumDistance
        ]

        if validEndPositions:
            return random.choice(validEndPositions)
        else:
            return max(startPositions, key=lambda pos: float(np.linalg.norm(np.array(pos) - np.array(self.start))))

        
    # UI rendering is handled in UI modules (VisualizationWindow/DisplayTools).
