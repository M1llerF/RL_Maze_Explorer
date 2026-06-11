from __future__ import annotations

from typing import Any, Dict

from environment.entities import normalize_enemy_behavior


class BaseTool:
    name: str = ""

    def apply(self, frame: Any, y: int, x: int) -> None:
        """Apply tool to a single cell at (y,x)."""
        raise NotImplementedError


class WallTool(BaseTool):
    name = "Wall"

    def apply(self, frame: Any, y: int, x: int) -> None:
        frame.gridData[y][x] = 1


class EraseTool(BaseTool):
    name = "Erase"

    def apply(self, frame: Any, y: int, x: int) -> None:
        frame.gridData[y][x] = 0
        frame.entities = [e for e in frame.entities if e.get("position") != [y, x]]


class StartTool(BaseTool):
    name = "Start"

    def apply(self, frame: Any, y: int, x: int) -> None:
        if frame.gridData[y][x] == 0:
            frame.start = (y, x)


class EndTool(BaseTool):
    name = "End"

    def apply(self, frame: Any, y: int, x: int) -> None:
        if frame.gridData[y][x] == 0:
            frame.end = (y, x)


class EnemyTool(BaseTool):
    name = "Enemy"

    def apply(self, frame: Any, y: int, x: int) -> None:
        if frame.gridData[y][x] != 0:
            return
        if (y, x) == tuple(frame.start) or (y, x) == tuple(frame.end):
            return
        existing = [e for e in frame.entities if e.get("position") == [y, x]]
        if existing:
            # Toggle off — remove enemy at this cell
            frame.entities = [e for e in frame.entities if e.get("position") != [y, x]]
        else:
            behavior_var = getattr(frame, "_enemyBehaviorVar", None)
            kind = behavior_var.get() if behavior_var is not None else "stationary"
            frame.entities = list(frame.entities) + [{
                "type": "enemy",
                "position": [y, x],
                "damage": 20.0,
                "alive": True,
                "behavior": normalize_enemy_behavior({"kind": kind}),
            }]


def _buildRegistry() -> Dict[str, BaseTool]:
    tools = [WallTool(), EraseTool(), StartTool(), EndTool(), EnemyTool()]
    return {t.name: t for t in tools}


TOOLREGISTRY: Dict[str, BaseTool] = _buildRegistry()
