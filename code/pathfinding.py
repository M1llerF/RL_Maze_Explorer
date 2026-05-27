import heapq
import numpy as np
from typing import Any
from collections import deque

class Pathfinding:
    @staticmethod
    def aStarSearch(maze: Any, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            return float(np.linalg.norm(np.array(a) - np.array(b)))
        
        def getNeighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
            neighbors: list[tuple[int, int]] = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                x2, y2 = pos[0] + dx, pos[1] + dy
                if 0 <= x2 < maze.height and 0 <= y2 < maze.width and maze.isValidPosition(None, x2, y2):
                    neighbors.append((x2, y2))
            return neighbors

        openList: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(openList, (0, start))
        cameFrom: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        costSoFar: dict[tuple[int, int], int] = {start: 0}
        
        while openList:
            _, current = heapq.heappop(openList)
            
            if current == goal:
                break
            
            for neighbor in getNeighbors(current):
                newCost = costSoFar[current] + 1
                if neighbor not in costSoFar or newCost < costSoFar[neighbor]:
                    costSoFar[neighbor] = newCost
                    priority = newCost + heuristic(goal, neighbor)
                    heapq.heappush(openList, (priority, neighbor))
                    cameFrom[neighbor] = current

        # Reconstruct path if reachable; otherwise return empty list
        if goal not in cameFrom:
            return []
        path: list[tuple[int, int]] = []
        current = goal
        while current is not None and current != start:
            path.append(current)
            current = cameFrom.get(current)
        if current == start:
            path.append(start)
            path.reverse()
            return path
        # Fallback: unreachable
        return []

    @staticmethod
    def bfsShortestPathGrid(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
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
