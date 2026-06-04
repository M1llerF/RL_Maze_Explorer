import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from typing import Any, cast

class DisplayTools:
    @staticmethod
    def createFrame(parent: Any, controller: Any, title: str) -> tk.Frame:
        frame = tk.Frame(parent)
        ttk.Label(frame, text=title, font=("TkDefaultFont", 20)).pack(pady=10, padx=10)
        return frame

    @staticmethod
    def loadProfiles(profileManager: Any, listbox: Any) -> None:
        profiles = profileManager.listProfiles()
        listbox.delete(0, tk.END)
        for profile in profiles:
            listbox.insert(tk.END, profile)

    @staticmethod
    def displayHeatmap(
        canvas: Any,
        maze: list[list[int]] | None,
        start: tuple[int, int],
        end: tuple[int, int],
        heatmapData: dict[tuple[int, int], int],
    ) -> None:
        # Clear the canvas
        canvas.delete("all")

        if maze is None:
            return

        mazeWidth = len(maze[0])
        mazeHeight = len(maze)
        cellWidth = canvas.winfo_width() / mazeWidth
        cellHeight = canvas.winfo_height() / mazeHeight

        heatmap = np.zeros((mazeHeight, mazeWidth))
        for (x, y), count in heatmapData.items():
            heatmap[x, y] = count

        maxHeat = heatmap.max() if heatmap.max() > 0 else 1  # Avoid division by zero
        cmap = plt.get_cmap("Reds")

        for y in range(mazeHeight):
            for x in range(mazeWidth):
                if maze[y][x] == 1:
                    canvas.create_rectangle(x * cellWidth, y * cellHeight,
                                                         (x + 1) * cellWidth, (y + 1) * cellHeight,
                                                         fill="black")
                else:
                    heatValue = heatmap[y, x] / maxHeat
                    if heatValue > 0:
                        color = mcolors.to_hex(cast(Any, cmap(heatValue)))
                        canvas.create_rectangle(x * cellWidth, y * cellHeight,
                                                             (x + 1) * cellWidth, (y + 1) * cellHeight,
                                                             fill=color, outline=color)

        canvas.create_rectangle(start[1] * cellWidth, start[0] * cellHeight,
                                             (start[1] + 1) * cellWidth, (start[0] + 1) * cellHeight,
                                             fill="blue")

        canvas.create_rectangle(end[1] * cellWidth, end[0] * cellHeight,
                                             (end[1] + 1) * cellWidth, (end[0] + 1) * cellHeight,
                                             fill="green")
