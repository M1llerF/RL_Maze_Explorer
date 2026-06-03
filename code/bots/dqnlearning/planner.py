from __future__ import annotations

from typing import Any, cast

from pathfinding import Pathfinding


class WarmupPlanner:
    """
    D*-Lite / frontier-based planner for warmup transition collection.

    The bot owns knownOpen, knownWalls, and seenGoals and passes live references
    at construction time. The planner reads them on every nextAction call but never
    modifies them. Wall discovery is signaled via onWallDiscovered().
    """

    _FRONTIER_REFRESH_INTERVAL: int = 6

    def __init__(
        self,
        maze: Any,
        profileName: str,
        tools: Any,
        knownOpen: set[tuple[int, int]],
        knownWalls: set[tuple[int, int]],
        seenGoals: set[tuple[int, int]],
    ) -> None:
        self._maze = maze
        self._profileName = profileName
        self._tools = tools
        # Live references — the bot mutates these; the planner only reads them.
        self._knownOpen = knownOpen
        self._knownWalls = knownWalls
        self._seenGoals = seenGoals
        self._planActions: list[int] = []
        self._planIndex: int = 0
        self._dstarPlanner: Any = None
        self._dstarGoal: tuple[int, int] | None = None
        self._dstarKnownBlocked: set[tuple[int, int]] = set()
        self._dirty: bool = True
        self._stepsSinceRefresh: int = self._FRONTIER_REFRESH_INTERVAL

    def nextAction(self, current: tuple[int, int]) -> int | None:
        """Return a valid action toward the goal, or None if no plan is available."""
        goal = cast(tuple[int, int], self._maze.end)
        goalKnown = goal in self._seenGoals
        for _ in range(4):
            if goalKnown:
                action = self._nextFromDstar(current, goal)
                if action is not None:
                    nextPos = self._tools.calculateNextPosition(current, action)
                    if self._maze.isValidPosition(self._profileName, nextPos[0], nextPos[1]):
                        return action
                    continue

            shouldRefresh = (
                self._planIndex >= len(self._planActions)
                or self._dirty
                or self._stepsSinceRefresh >= self._FRONTIER_REFRESH_INTERVAL
            )
            if shouldRefresh:
                self._planActions = self._tools.getDynamicWarmupActions(
                    current,
                    goal,
                    maxSteps=64,
                    knownOpen=set(self._knownOpen),
                    knownWalls=set(self._knownWalls),
                    seenGoals=set(self._seenGoals),
                )
                self._planIndex = 0
                self._stepsSinceRefresh = 0
            if self._planIndex < len(self._planActions):
                action = int(self._planActions[self._planIndex])
                self._planIndex += 1
                self._stepsSinceRefresh += 1
                nextPos = self._tools.calculateNextPosition(current, action)
                if self._maze.isValidPosition(self._profileName, nextPos[0], nextPos[1]):
                    return action
                continue
            break
        return None

    def onWallDiscovered(self) -> None:
        """Invalidate the current plan after a wall is encountered."""
        self._planActions = []
        self._planIndex = 0
        self._stepsSinceRefresh = self._FRONTIER_REFRESH_INTERVAL
        self._dirty = True

    def reset(self) -> None:
        """Reset all planning state for a new episode."""
        self._planActions = []
        self._planIndex = 0
        self._dstarPlanner = None
        self._dstarGoal = None
        self._dstarKnownBlocked.clear()
        self._dirty = True
        self._stepsSinceRefresh = self._FRONTIER_REFRESH_INTERVAL

    def _nextFromDstar(self, current: tuple[int, int], goal: tuple[int, int]) -> int | None:
        if self._dstarPlanner is None or self._dirty or self._dstarGoal != goal:
            self._dstarPlanner = Pathfinding.DStarLitePlanner(
                height=int(self._maze.height),
                width=int(self._maze.width),
                start=current,
                goal=goal,
                blocked=set(self._knownWalls),
            )
            self._dstarGoal = goal
            self._dstarKnownBlocked = set(self._knownWalls)
            self._dirty = False
        else:
            self._dstarPlanner.moveStart(current)
            newBlocked = set(self._knownWalls) - self._dstarKnownBlocked
            if newBlocked:
                self._dstarPlanner.updateBlockedCells(newBlocked)
                self._dstarKnownBlocked.update(newBlocked)

        nxt = self._dstarPlanner.nextStep()
        if nxt is None:
            return None
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        return {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}.get(delta)
