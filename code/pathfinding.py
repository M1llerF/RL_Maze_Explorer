import heapq
import numpy as np
from typing import Any
from collections import deque

class Pathfinding:
    class _DStarLitePlanner:
        def __init__(
            self,
            height: int,
            width: int,
            start: tuple[int, int],
            goal: tuple[int, int],
            blocked: set[tuple[int, int]] | None = None,
        ) -> None:
            self.height = int(height)
            self.width = int(width)
            self.start = start
            self.goal = goal
            self.km = 0.0
            self._lastStart = start
            self._infinity = float("inf")
            self._blocked: set[tuple[int, int]] = set(blocked or set())
            self._g: dict[tuple[int, int], float] = {}
            self._rhs: dict[tuple[int, int], float] = {goal: 0.0}
            self._open: list[tuple[float, float, tuple[int, int]]] = []
            self._openBest: dict[tuple[int, int], tuple[float, float]] = {}
            self._push(goal)

        def _inBounds(self, node: tuple[int, int]) -> bool:
            row, col = node
            return 0 <= row < self.height and 0 <= col < self.width

        def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
            return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

        def _neighbors(self, node: tuple[int, int]) -> list[tuple[int, int]]:
            row, col = node
            candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            return [c for c in candidates if self._inBounds(c) and c not in self._blocked]

        def _gScore(self, node: tuple[int, int]) -> float:
            return self._g.get(node, self._infinity)

        def _rhsScore(self, node: tuple[int, int]) -> float:
            return self._rhs.get(node, self._infinity)

        def _cost(self, _a: tuple[int, int], _b: tuple[int, int]) -> float:
            return 1.0

        def _key(self, node: tuple[int, int]) -> tuple[float, float]:
            best = min(self._gScore(node), self._rhsScore(node))
            return (best + self._heuristic(self.start, node) + self.km, best)

        def _push(self, node: tuple[int, int]) -> None:
            first, second = self._key(node)
            self._openBest[node] = (first, second)
            heapq.heappush(self._open, (first, second, node))

        def _topKey(self) -> tuple[float, float]:
            while self._open:
                first, second, node = self._open[0]
                best = self._openBest.get(node)
                if best == (first, second) and (first, second) == self._key(node):
                    return first, second
                heapq.heappop(self._open)
            return self._infinity, self._infinity

        def _updateVertex(self, node: tuple[int, int]) -> None:
            if node != self.goal:
                self._rhs[node] = min(
                    (self._cost(node, succ) + self._gScore(succ) for succ in self._neighbors(node)),
                    default=self._infinity,
                )
            if self._gScore(node) != self._rhsScore(node):
                self._push(node)

        def moveStart(self, newStart: tuple[int, int]) -> None:
            if newStart != self.start:
                self.km += self._heuristic(self._lastStart, newStart)
                self._lastStart = newStart
                self.start = newStart

        def updateBlockedCells(self, newBlocked: set[tuple[int, int]]) -> None:
            changed = [node for node in newBlocked if node not in self._blocked and self._inBounds(node)]
            if not changed:
                return
            for node in changed:
                self._blocked.add(node)
                self._g.pop(node, None)
                self._rhs.pop(node, None)
                self._updateVertex(node)
                for neighbor in self._neighbors(node):
                    self._updateVertex(neighbor)

        def computeShortestPath(self) -> None:
            maxIterations = max(64, self.height * self.width * 4)
            iterations = 0
            while self._topKey() < self._key(self.start) or self._rhsScore(self.start) != self._gScore(self.start):
                iterations += 1
                if iterations > maxIterations or not self._open:
                    break
                oldFirst, oldSecond, node = heapq.heappop(self._open)
                best = self._openBest.get(node)
                if best is not None and best != (oldFirst, oldSecond):
                    continue
                self._openBest.pop(node, None)
                oldKey = (oldFirst, oldSecond)
                newKey = self._key(node)
                if oldKey < newKey:
                    self._push(node)
                elif self._gScore(node) > self._rhsScore(node):
                    self._g[node] = self._rhsScore(node)
                    for pred in self._neighbors(node):
                        self._updateVertex(pred)
                else:
                    self._g[node] = self._infinity
                    self._updateVertex(node)
                    for pred in self._neighbors(node):
                        self._updateVertex(pred)

        def nextStep(self) -> tuple[int, int] | None:
            self.computeShortestPath()
            if self._gScore(self.start) == self._infinity:
                return None
            successors = self._neighbors(self.start)
            if not successors:
                return None
            return min(
                successors,
                key=lambda succ: (self._cost(self.start, succ) + self._gScore(succ), self._heuristic(succ, self.goal), succ[0], succ[1]),
            )

    @staticmethod
    def _aStarOnKnownWalls(
        height: int,
        width: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        knownWalls: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if start == goal:
            return [start]
        if start in knownWalls or goal in knownWalls:
            return []
        openList: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(openList, (0.0, start))
        cameFrom: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        gScore: dict[tuple[int, int], int] = {start: 0}
        while openList:
            _, current = heapq.heappop(openList)
            if current == goal:
                break
            row, col = current
            for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                    continue
                if nxt in knownWalls:
                    continue
                tentative = gScore[current] + 1
                if nxt not in gScore or tentative < gScore[nxt]:
                    gScore[nxt] = tentative
                    priority = float(tentative + abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1]))
                    heapq.heappush(openList, (priority, nxt))
                    cameFrom[nxt] = current
        if goal not in cameFrom:
            return []
        out: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = goal
        while cur is not None:
            out.append(cur)
            cur = cameFrom[cur]
        out.reverse()
        return out

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

    @staticmethod
    def dStarLiteSearch(
        maze: Any,
        start: tuple[int, int],
        goal: tuple[int, int],
        knownWalls: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        planner = Pathfinding._DStarLitePlanner(
            height=int(getattr(maze, "height", 0)),
            width=int(getattr(maze, "width", 0)),
            start=start,
            goal=goal,
            blocked=set(knownWalls or set()),
        )
        path = [start]
        current = start
        seen = {start}
        for _ in range(max(1, int(getattr(maze, "height", 0)) * int(getattr(maze, "width", 0)))):
            if current == goal:
                return path
            planner.moveStart(current)
            nxt = planner.nextStep()
            if nxt is None or nxt in seen:
                return []
            path.append(nxt)
            seen.add(nxt)
            current = nxt
        return []

    @staticmethod
    def dStarLiteFrontierWarmupActions(
        maze: Any,
        start: tuple[int, int],
        goal: tuple[int, int],
        maxSteps: int | None = None,
        knownOpen: set[tuple[int, int]] | None = None,
        knownWalls: set[tuple[int, int]] | None = None,
        seenGoals: set[tuple[int, int]] | None = None,
    ) -> list[int]:
        seenGoalCells = set(seenGoals or set())
        knownWallCells = set(knownWalls or set())
        # Only target the final goal after it is observable in current knowledge.
        if goal in seenGoalCells:
            path = Pathfinding.dStarLiteSearch(maze, start, goal, knownWalls=knownWallCells)
            if len(path) >= 2:
                actions: list[int] = []
                for i in range(len(path) - 1):
                    a = path[i]
                    b = path[i + 1]
                    delta = (b[0] - a[0], b[1] - a[1])
                    action = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}.get(delta)
                    if action is None:
                        return []
                    actions.append(action)
                    if maxSteps is not None and len(actions) >= int(maxSteps):
                        break
                return actions
        # Fallback frontier: greedy nearest unknown-adjacent open cell from known open set.
        knownOpenCells = set(knownOpen or set())
        if start not in knownOpenCells:
            knownOpenCells.add(start)
        frontier = []
        for cell in knownOpenCells:
            for n in ((cell[0]-1, cell[1]), (cell[0]+1, cell[1]), (cell[0], cell[1]-1), (cell[0], cell[1]+1)):
                if 0 <= n[0] < maze.height and 0 <= n[1] < maze.width and n not in knownOpenCells and n not in knownWallCells:
                    frontier.append(cell)
                    break
        if not frontier:
            return []
        target = min(frontier, key=lambda c: abs(c[0]-start[0]) + abs(c[1]-start[1]))
        path2 = Pathfinding._aStarOnKnownWalls(
            height=int(maze.height),
            width=int(maze.width),
            start=start,
            goal=target,
            knownWalls=knownWallCells,
        )
        if len(path2) < 2:
            return []
        actions2: list[int] = []
        for i in range(len(path2)-1):
            a, b = path2[i], path2[i+1]
            action = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}.get((b[0]-a[0], b[1]-a[1]))
            if action is None:
                return []
            actions2.append(action)
            if maxSteps is not None and len(actions2) >= int(maxSteps):
                break
        return actions2
