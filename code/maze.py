import numpy as np
import random
from typing import Any


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
        
    def setupSimpleMaze(self) -> None:
        # Randomly adjust width and height
        # self.width = random.randint(self.width, self.width + 2)
        # self.height = random.randint(self.height, self.height + 2)

        # Initialize grid with walls
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]

        def dfsIterative(x: int, y: int, algorithmType: int) -> None:
            stack = [(x, y)]
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

            while stack:
                cx, cy = stack.pop()
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = cx + 2 * dx, cy + 2 * dy
                    if 1 <= nx < self.height - 1 and 1 <= ny < self.width - 1 and self.grid[nx][ny] == 1:
                        self.grid[cx + dx][cy + dy] = 0
                        self.grid[nx][ny] = 0
                        if algorithmType == 1:
                            stack.append((cx, cy))
                        stack.append((nx, ny))
                        break

        # Randomly choose a starting position
        startX = random.randrange(1, self.height - 1, 2)
        startY = random.randrange(1, self.width - 1, 2)
        self.grid[startX][startY] = 0

        # Randomly choose a algorithm type
        algorithmType = random.randint(1, 2)
        dfsIterative(startX, startY, algorithmType)

        # Collect all path positions
        startPositions = [(x, y) for x in range(1, self.height - 1) for y in range(1, self.width - 1) if self.grid[x][y] == 0]
        self.start = random.choice(startPositions)

        # Ensure the goal is at least minimum_distance away from the start
        self.end = self.getFarthestValidEndPosition(startPositions)

        self.setStart(*self.start)
        self.setGoal(*self.end)

    def getState(self) -> MazeState:
        """Return a serializable snapshot of the current maze state."""
        return {
            'width': self.width,
            'height': self.height,
            'grid': [row[:] for row in self.grid],
            'start': tuple(self.start) if self.start is not None else None,
            'end': tuple(self.end) if self.end is not None else None,
        }

    def setState(self, state: MazeState) -> None:
        """Load a snapshot produced by get_state()."""
        self.width = int(state['width'])
        self.height = int(state['height'])
        self.grid = [row[:] for row in state['grid']]
        self.start = tuple(state['start']) if state['start'] is not None else None
        self.end = tuple(state['end']) if state['end'] is not None else None

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
