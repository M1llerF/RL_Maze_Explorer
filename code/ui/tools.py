from __future__ import annotations

from typing import Any, Dict


class BaseTool:
    name: str = ""

    def apply(self, frame: Any, y: int, x: int) -> None:
        """Apply tool to a single cell at (y,x)."""
        raise NotImplementedError


class WallTool(BaseTool):
    name = "Wall"

    def apply(self, frame: Any, y: int, x: int) -> None:
        frame.gridData[y][x] = 1


class PathTool(BaseTool):
    name = "Path"

    def apply(self, frame: Any, y: int, x: int) -> None:
        frame.gridData[y][x] = 0


class EraseTool(BaseTool):
    name = "Erase"

    def apply(self, frame: Any, y: int, x: int) -> None:
        frame.gridData[y][x] = 0


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


def _buildRegistry() -> Dict[str, BaseTool]:
    tools = [WallTool(), PathTool(), EraseTool(), StartTool(), EndTool()]
    return {t.name: t for t in tools}


TOOLREGISTRY: Dict[str, BaseTool] = _buildRegistry()

