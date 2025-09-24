from __future__ import annotations

from typing import Dict, Callable


class BaseTool:
    name: str = ""

    def apply(self, frame, y: int, x: int) -> None:
        """Apply tool to a single cell at (y,x)."""
        raise NotImplementedError


class WallTool(BaseTool):
    name = "Wall"

    def apply(self, frame, y: int, x: int) -> None:
        frame.grid_data[y][x] = 1


class PathTool(BaseTool):
    name = "Path"

    def apply(self, frame, y: int, x: int) -> None:
        frame.grid_data[y][x] = 0


class EraseTool(BaseTool):
    name = "Erase"

    def apply(self, frame, y: int, x: int) -> None:
        frame.grid_data[y][x] = 0


class StartTool(BaseTool):
    name = "Start"

    def apply(self, frame, y: int, x: int) -> None:
        if frame.grid_data[y][x] == 0:
            frame.start = (y, x)


class EndTool(BaseTool):
    name = "End"

    def apply(self, frame, y: int, x: int) -> None:
        if frame.grid_data[y][x] == 0:
            frame.end = (y, x)


def _build_registry() -> Dict[str, BaseTool]:
    tools = [WallTool(), PathTool(), EraseTool(), StartTool(), EndTool()]
    return {t.name: t for t in tools}


TOOL_REGISTRY: Dict[str, BaseTool] = _build_registry()

