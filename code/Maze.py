import numpy as np
import random
class Maze:
    def __init__(self, width, height, start=None, end=None):
        """
        Initialize the maze with given dimensions and optionally set start and end points.
        """
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.start = start
        self.end = end
        self.minimum_distance = max(width, height) // 2
        self.setup_simple_maze()
        # Do not create a matplotlib figure here; the app uses Tk canvas.
        # Leaving interactive figure creation disabled prevents stray empty windows.
    
    def is_valid_position(self, bot_name, x, y):
        """
        Check if a position is valid (within bounds and not a wall).
        Pure check: no side effects.
        """
        return 0 <= x < self.height and 0 <= y < self.width and self.grid[x][y] == 0

    def set_wall(self, x, y):
        """
        Set a wall at the specified position.
        Raise a error if the position is out of bounds.
        """
        if 0 <= x < self.height and 0 <= y < self.width:
            self.grid[x][y] = 1
        else:
            raise ValueError("Position out of maze bounds")
    
    def get_start(self):
        """
        Get the start position of the maze.
        """
        return self.start
    
    def set_start(self, x, y):
        if self.is_valid_position(None, x, y):
            self.start = (x, y)
        else:
            raise ValueError("Invalid start position")
    
    def set_goal(self, x, y):
        """
        Set the goal position of the maze if it's valid.
        Raise a error if the position is invalid.
        """
        if self.is_valid_position(None, x, y):
            self.end = (x, y)
        else:
            raise ValueError("Invalid goal position")
        
    def setup_simple_maze(self):
        # Randomly adjust width and height
        # self.width = random.randint(self.width, self.width + 2)
        # self.height = random.randint(self.height, self.height + 2)

        # Initialize grid with walls
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]

        def dfs_iterative(x, y, algorithm_type):
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
                        if algorithm_type == 1:
                            stack.append((cx, cy))
                        stack.append((nx, ny))
                        break

        # Randomly choose a starting position
        start_x = random.randrange(1, self.height - 1, 2)
        start_y = random.randrange(1, self.width - 1, 2)
        self.grid[start_x][start_y] = 0

        # Randomly choose a algorithm type
        algorithm_type = random.randint(1, 2)
        dfs_iterative(start_x, start_y, algorithm_type)

        # Collect all path positions
        start_positions = [(x, y) for x in range(1, self.height - 1) for y in range(1, self.width - 1) if self.grid[x][y] == 0]
        self.start = random.choice(start_positions)

        # Ensure the goal is at least minimum_distance away from the start
        self.end = self.get_farthest_valid_end_position(start_positions)

        self.set_start(*self.start)
        self.set_goal(*self.end)

    def get_state(self):
        """Return a serializable snapshot of the current maze state."""
        return {
            'width': self.width,
            'height': self.height,
            'grid': [row[:] for row in self.grid],
            'start': tuple(self.start) if self.start is not None else None,
            'end': tuple(self.end) if self.end is not None else None,
        }

    def set_state(self, state):
        """Load a snapshot produced by get_state()."""
        self.width = int(state['width'])
        self.height = int(state['height'])
        self.grid = [row[:] for row in state['grid']]
        self.start = tuple(state['start']) if state['start'] is not None else None
        self.end = tuple(state['end']) if state['end'] is not None else None

    def get_farthest_valid_end_position(self, start_positions):
        """
        Get the farthest valid end position that is at least minimum_distance away from the start.
        """
        valid_end_positions = [
            pos for pos in start_positions if np.linalg.norm(np.array(pos) - np.array(self.start)) >= self.minimum_distance
        ]

        if valid_end_positions:
            return random.choice(valid_end_positions)
        else:
            return max(start_positions, key=lambda pos: np.linalg.norm(np.array(pos) - np.array(self.start)))

        
    # UI rendering is handled in UI modules (VisualizationWindow/DisplayTools).
