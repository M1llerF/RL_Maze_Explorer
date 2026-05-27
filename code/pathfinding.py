import heapq
import numpy as np
from typing import Any
from collections import deque

class Pathfinding:
    @staticmethod
    def a_star_search(maze: Any, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            return float(np.linalg.norm(np.array(a) - np.array(b)))
        
        def get_neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
            neighbors: list[tuple[int, int]] = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                x2, y2 = pos[0] + dx, pos[1] + dy
                if 0 <= x2 < maze.height and 0 <= y2 < maze.width and maze.is_valid_position(None, x2, y2):
                    neighbors.append((x2, y2))
            return neighbors

        open_list: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_list, (0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], int] = {start: 0}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            
            if current == goal:
                break
            
            for neighbor in get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(goal, neighbor)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current

        # Reconstruct path if reachable; otherwise return empty list
        if goal not in came_from:
            return []
        path: list[tuple[int, int]] = []
        current = goal
        while current is not None and current != start:
            path.append(current)
            current = came_from.get(current)
        if current == start:
            path.append(start)
            path.reverse()
            return path
        # Fallback: unreachable
        return []

    @staticmethod
    def bfs_shortest_path_grid(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Compute shortest path on a 4-connected grid using BFS.
        grid: 2D list where 0=open, 1=wall. start/goal are (y,x).
        Returns list of (y,x) from start to goal inclusive, or [] if unreachable
        """
        if not grid:
            return []
        h = len(grid)
        w = len(grid[0]) if h else 0
        sy, sx = start
        ey, ex = goal
        if not (0 <= sy < h and 0 <= sx < w and 0 <= ey < h and 0 <= ex < w):
            return []
        if grid[sy][sx] == 1 or grid[ey][ex] == 1:
            return []
        q: deque[tuple[int, int]] = deque()
        q.append((sy, sx))
        prev: dict[tuple[int, int], tuple[int, int] | None] = {(sy, sx): None}
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            y, x = q.popleft()
            if (y, x) == (ey, ex):
                break
            for dy, dx in moves:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == 0 and (ny, nx) not in prev:
                    prev[(ny, nx)] = (y, x)
                    q.append((ny, nx))
        if (ey, ex) not in prev:
            return []
        cur: tuple[int, int] | None = (ey, ex)
        out: list[tuple[int, int]] = []
        while cur is not None:
            out.append(cur)
            cur = prev[cur]
        out.reverse()
        return out
