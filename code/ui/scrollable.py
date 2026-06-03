import tkinter as tk
from tkinter import ttk
from typing import Any


class VerticalScrolledFrame(ttk.Frame):
    """A reusable vertically scrollable content host."""

    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)
        self._windowId = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.content.bind("<Configure>", self._onContentConfigure)
        self.canvas.bind("<Configure>", self._onCanvasConfigure)

        for widget in (self.canvas, self.content):
            widget.bind("<MouseWheel>", self._onMouseWheel)
            widget.bind("<Button-4>", self._onMouseWheel)
            widget.bind("<Button-5>", self._onMouseWheel)

    def _onContentConfigure(self, _event: Any) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _onCanvasConfigure(self, event: Any) -> None:
        self.canvas.itemconfigure(self._windowId, width=event.width)

    def _onMouseWheel(self, event: Any) -> str:
        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", None)
        if delta > 0 or num == 4:
            self.canvas.yview_scroll(-1, "units")
        else:
            self.canvas.yview_scroll(1, "units")
        return "break"
