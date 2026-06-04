import heapq
import numpy as np
from typing import Any
from collections import deque
from environment.traversal import TraversalPolicy, build_traversal_policy

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DELTA_TO_ACTION = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}


class Pathfinding:
    class DStarLitePlanner:
        """
        Incremental partial-knowledge planner.  Unknown cells are optimistically
        treated as open; only confirmed walls are blocked.  Replans via A* on
        every nextStep call — at ≤400 cells this is sub-millisecond.
        """

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
            self._blocked: set[tuple[int, int]] = set(blocked or ())

        def moveStart(self, newStart: tuple[int, int]) -> None:
            self.start = newStart

        def updateBlockedCells(self, newBlocked: set[tuple[int, int]]) -> None:
            self._blocked.update(newBlocked)

        def nextStep(self) -> tuple[int, int] | None:
            path = Pathfinding._aStarPartialKnowledge(
                self.height, self.width, self.start, self.goal, self._blocked
            )
            return path[1] if len(path) >= 2 else None

    # ------------------------------------------------------------------
    # Core A* on partial knowledge
    # ------------------------------------------------------------------

    @staticmethod
    def _aStarPartialKnowledge(
        height: int,
        width: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        knownWalls: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        A* from start to goal.  Unknown cells are treated as traversable;
        only cells in knownWalls are blocked.  Returns [] when unreachable.
        """
        if start == goal:
            return [start]
        if start in knownWalls or goal in knownWalls:
            return []

        h, w = height, width
        gr, gc = goal

        def heur(r: int, c: int) -> int:
            return abs(gr - r) + abs(gc - c)

        heap: list[tuple[int, int, tuple[int, int]]] = [
            (heur(start[0], start[1]), 0, start)
        ]
        g: dict[tuple[int, int], int] = {start: 0}
        came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while heap:
            _, cost, cur = heapq.heappop(heap)
            if cur == goal:
                path: list[tuple[int, int]] = []
                node: tuple[int, int] | None = cur
                while node is not None:
                    path.append(node)
                    node = came[node]
                path.reverse()
                return path
            if cost > g.get(cur, 10**9):
                continue
            r, c = cur
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                nb = (nr, nc)
                if not (0 <= nr < h and 0 <= nc < w) or nb in knownWalls:
                    continue
                ng = cost + 1
                if ng < g.get(nb, 10**9):
                    g[nb] = ng
                    came[nb] = cur
                    heapq.heappush(heap, (ng + heur(nr, nc), ng, nb))
        return []

    # ------------------------------------------------------------------
    # Frontier exploration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unknownComponentSize(
        height: int,
        width: int,
        seed: tuple[int, int],
        knownOpen: set[tuple[int, int]],
        knownWalls: set[tuple[int, int]],
        limit: int,
    ) -> int:
        """
        BFS count of unknown cells reachable from seed through unknown space.
        Stops early once count reaches limit (we only need to know ≥ limit).
        """
        if seed in knownOpen or seed in knownWalls:
            return 0
        visited: set[tuple[int, int]] = {seed}
        queue: deque[tuple[int, int]] = deque([seed])
        count = 0
        while queue and count < limit:
            r, c = queue.popleft()
            count += 1
            for dr, dc in _DIRS:
                nb = (r + dr, c + dc)
                if (
                    0 <= nb[0] < height
                    and 0 <= nb[1] < width
                    and nb not in visited
                    and nb not in knownOpen
                    and nb not in knownWalls
                ):
                    visited.add(nb)
                    queue.append(nb)
        return count

    @staticmethod
    def _bestFrontierTarget(
        height: int,
        width: int,
        start: tuple[int, int],
        knownOpen: set[tuple[int, int]],
        knownWalls: set[tuple[int, int]],
        pocketThreshold: int = 3,
    ) -> tuple[int, int] | None:
        """
        Find the closest frontier cell (knownOpen adjacent to unknown space)
        whose unknown component is large enough to plausibly contain the goal.

        Cells whose unknown neighbours all belong to isolated pockets (component
        size < pocketThreshold) are skipped.  If every frontier is a pocket the
        nearest one is returned anyway, guaranteeing progress toward 100% coverage.
        """
        h, w = height, width
        frontier: list[tuple[int, int, int, tuple[int, int]]] = []

        for cell in knownOpen:
            r, c = cell
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                nb = (nr, nc)
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and nb not in knownOpen
                    and nb not in knownWalls
                ):
                    dist = abs(r - start[0]) + abs(c - start[1])
                    frontier.append((dist, r, c, cell))
                    break

        if not frontier:
            return None

        frontier.sort()
        fallback = frontier[0][3]

        for _, r, c, cell in frontier:
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                nb = (nr, nc)
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and nb not in knownOpen
                    and nb not in knownWalls
                    and Pathfinding._unknownComponentSize(
                        h, w, nb, knownOpen, knownWalls, pocketThreshold
                    ) >= pocketThreshold
                ):
                    return cell

        return fallback

    # ------------------------------------------------------------------
    # Public API (names kept for backward compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _aStarOnKnownWalls(
        height: int,
        width: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        knownWalls: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        return Pathfinding._aStarPartialKnowledge(height, width, start, goal, knownWalls)

    @staticmethod
    def aStarSearch(
        maze: Any,
        start: tuple[int, int],
        goal: tuple[int, int],
        traversal: TraversalPolicy | None = None,
    ) -> list[tuple[int, int]]:
        policy = traversal or build_traversal_policy(maze)

        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            return float(np.linalg.norm(np.array(a) - np.array(b)))

        def getNeighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
            neighbors: list[tuple[int, int]] = []
            for dx, dy in _DIRS:
                x2, y2 = pos[0] + dx, pos[1] + dy
                if 0 <= x2 < maze.height and 0 <= y2 < maze.width and policy.is_valid_position((x2, y2)):
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
        return []

    @staticmethod
    def bfsShortestPathGrid(
        grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
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
        while q:
            y, x = q.popleft()
            if (y, x) == (ey, ex):
                break
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
        return Pathfinding._aStarPartialKnowledge(
            height=int(getattr(maze, "height", 0)),
            width=int(getattr(maze, "width", 0)),
            start=start,
            goal=goal,
            knownWalls=set(knownWalls or ()),
        )

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
        seenGoalCells = set(seenGoals or ())
        knownWallCells = set(knownWalls or ())
        knownOpenCells = set(knownOpen or ())
        if start not in knownOpenCells:
            knownOpenCells.add(start)

        height = int(getattr(maze, "height", 0))
        width = int(getattr(maze, "width", 0))

        if goal in seenGoalCells:
            path = Pathfinding._aStarPartialKnowledge(height, width, start, goal, knownWallCells)
        else:
            target = Pathfinding._bestFrontierTarget(height, width, start, knownOpenCells, knownWallCells)
            if target is None:
                return []
            path = Pathfinding._aStarPartialKnowledge(height, width, start, target, knownWallCells)

        if len(path) < 2:
            return []

        actions: list[int] = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            action = _DELTA_TO_ACTION.get((b[0] - a[0], b[1] - a[1]))
            if action is None:
                return []
            actions.append(action)
            if maxSteps is not None and len(actions) >= int(maxSteps):
                break
        return actions
