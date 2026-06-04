# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np

from displayTools import DisplayTools
from rewardGrapher import RewardGrapher
from services.visualization_service import VisualizationSnapshot, VisualizationService
from ui.event_bus import PROFILE_SAVED, PROFILE_DELETED
from visualizationStrategy import DefaultVisualizationStrategy


class VisualizationWindow(tk.Toplevel):
    def __init__(self, parent: Any, gameEnv: Any, profileName: str, profileIndex: int) -> None:
        super().__init__(parent)
        self.gameEnv = gameEnv
        self.profileName = profileName
        self.profileIndex = profileIndex
        self._visualizationService = VisualizationService(self.gameEnv.repository)
        self.title("Maze Visualization")
        self.geometry("600x600")

        banner = tk.Label(self, text="Training is PAUSED for this profile while this window is open.",
                          bg="#fff3cd", fg="#856404", anchor="center")
        banner.pack(fill=tk.X)

        self.canvas = tk.Canvas(self, width=500, height=500, bg="white")
        self.canvas.pack(pady=20)

        controls = tk.Frame(self)
        controls.pack(pady=5)
        tk.Label(controls, text="Steps per frame:").pack(side=tk.LEFT)
        # Default to one simulation step per frame so enemy movement is visible.
        self.stepsVar = tk.IntVar(value=1)
        self.stepsScale = tk.Scale(controls, from_=1, to=200, orient=tk.HORIZONTAL, variable=self.stepsVar, length=200)
        self.stepsScale.pack(side=tk.LEFT, padx=5)
        self.debugVar = tk.BooleanVar(value=False)
        self.debugCheck = ttk.Checkbutton(
            controls,
            text="Debug Prints",
            variable=self.debugVar,
            command=self._applyDebugToggle,
        )
        self.debugCheck.pack(side=tk.LEFT, padx=5)
        self.paused = False
        self.pauseBtn = ttk.Button(controls, text="Pause", command=self.togglePause)
        self.pauseBtn.pack(side=tk.LEFT, padx=5)

        self.afterId = None
        self.visualize = True
        self._attackFlashes: dict[tuple[int, int], int] = {}
        self.protocol("WM_DELETE_WINDOW", self.onClose)

        try:
            self.gameEnv.pauseTrainingFor(self.profileName)
        except Exception:
            pass
        self._applyDebugToggle()
        self._subscribeAttackEvents()

        self.updateVisualization()

    def updateVisualization(self) -> None:
        if not self.visualize:
            return
        bot = self.gameEnv.bots[self.profileIndex]
        if not self.paused:
            try:
                finished = bot.stepVisualization(maxSteps=self.stepsVar.get())
                if finished:
                    self.gameEnv.resetEnvironment(self.profileIndex)
            except Exception:
                pass
        snapshot = self._visualizationService.build_snapshot(self.profileName, bot)
        self.renderVisualizationSnapshot(snapshot)
        self.afterId = self.after(100, self.updateVisualization)

    def _subscribeAttackEvents(self) -> None:
        try:
            from environment.context import ENEMY_KILLED
            bot = self.gameEnv.bots[self.profileIndex]
            context = getattr(bot, "_context", None)
            if context is not None:
                context.on(ENEMY_KILLED, lambda **p: self._onEnemyKilled(p))
        except Exception:
            pass

    def _onEnemyKilled(self, payload: dict) -> None:
        pos = payload.get("position")
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            self._attackFlashes[(int(pos[0]), int(pos[1]))] = 5

    def togglePause(self) -> None:
        self.paused = not self.paused
        self.pauseBtn.configure(text="Resume" if self.paused else "Pause")

    def _applyDebugToggle(self) -> None:
        try:
            bot = self.gameEnv.bots[self.profileIndex]
        except Exception:
            return
        setter = getattr(bot, "setVisualizationDebug", None)
        if callable(setter):
            try:
                setter(bool(self.debugVar.get()))
            except Exception:
                pass

    def renderVisualizationSnapshot(self, snapshot: VisualizationSnapshot) -> None:
        self.canvas.delete("all")
        if snapshot.maze_grid is None or snapshot.bot_position is None:
            return
        grid = snapshot.maze_grid
        height = len(grid)
        width = len(grid[0]) if height else 0
        if height <= 0 or width <= 0:
            return
        cellWidth = self.canvas.winfo_width() / width
        cellHeight = self.canvas.winfo_height() / height
        heatmap = np.zeros((height, width))
        for (row, col), count in (snapshot.heatmap_data or {}).items():
            if 0 <= row < height and 0 <= col < width:
                heatmap[row, col] = count
        maxHeat = heatmap.max() if heatmap.max() > 0 else 1
        cmap = plt.get_cmap("Reds")
        for y in range(height):
            for x in range(width):
                if grid[y][x] == 1:
                    self.canvas.create_rectangle(x * cellWidth, y * cellHeight,
                                                 (x + 1) * cellWidth, (y + 1) * cellHeight,
                                                 fill="black")
                else:
                    heatValue = heatmap[y, x] / maxHeat
                    if heatValue > 0:
                        color = mcolors.to_hex(cmap(heatValue))
                        self.canvas.create_rectangle(x * cellWidth, y * cellHeight,
                                                     (x + 1) * cellWidth, (y + 1) * cellHeight,
                                                     fill=color, outline=color)
        for row, col in snapshot.path or []:
            self.canvas.create_rectangle(
                col * cellWidth,
                row * cellHeight,
                (col + 1) * cellWidth,
                (row + 1) * cellHeight,
                outline="#f4b400",
                width=2,
            )
        if snapshot.maze_start is not None:
            start = snapshot.maze_start
            self.canvas.create_rectangle(start[1] * cellWidth, start[0] * cellHeight,
                                         (start[1] + 1) * cellWidth, (start[0] + 1) * cellHeight,
                                         fill="blue")
        if snapshot.maze_end is not None:
            end = snapshot.maze_end
            self.canvas.create_rectangle(end[1] * cellWidth, end[0] * cellHeight,
                                         (end[1] + 1) * cellWidth, (end[0] + 1) * cellHeight,
                                         fill="green")
        for enemy in snapshot.enemies:
            er, ec = enemy.position
            x0, y0 = ec * cellWidth, er * cellHeight
            x1, y1 = (ec + 1) * cellWidth, (er + 1) * cellHeight
            if enemy.alive:
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="#e65100", outline="#bf360c", width=2)
                label = enemy.behavior_kind[0].upper()
                self.canvas.create_text(
                    (x0 + x1) / 2, (y0 + y1) / 2,
                    text=label, fill="white", font=("TkDefaultFont", max(6, int(min(cellWidth, cellHeight) * 0.45))),
                )
            else:
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="#616161", outline="#424242")
        # Attack flash: bright burst at killed-enemy positions, decays over 5 frames
        flashes = getattr(self, "_attackFlashes", {})
        expired = [pos for pos, ttl in flashes.items() if ttl <= 0]
        for pos in expired:
            del flashes[pos]
        for (fr, fc), ttl in list(flashes.items()):
            intensity = ttl / 5.0
            x0, y0 = fc * cellWidth, fr * cellHeight
            x1, y1 = (fc + 1) * cellWidth, (fr + 1) * cellHeight
            pad = max(1.0, min(cellWidth, cellHeight) * 0.08 * (1.0 - intensity))
            self.canvas.create_oval(
                x0 + pad, y0 + pad, x1 - pad, y1 - pad,
                fill="#ffeb3b", outline="#ff6f00", width=max(1, int(2 * intensity)),
            )
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            arm = min(cellWidth, cellHeight) * 0.35 * intensity
            self.canvas.create_line(cx - arm, cy, cx + arm, cy, fill="white", width=max(1, int(2 * intensity)))
            self.canvas.create_line(cx, cy - arm, cx, cy + arm, fill="white", width=max(1, int(2 * intensity)))
            flashes[(fr, fc)] -= 1

        botPosition = snapshot.bot_position
        self.canvas.create_oval(botPosition[1] * cellWidth, botPosition[0] * cellHeight,
                                (botPosition[1] + 1) * cellWidth, (botPosition[0] + 1) * cellHeight,
                                fill="red")

    def onClose(self) -> None:
        self.visualize = False
        if self.afterId is not None:
            self.after_cancel(self.afterId)
        try:
            bot = self.gameEnv.bots[self.profileIndex]
            setter = getattr(bot, "setVisualizationDebug", None)
            if callable(setter):
                setter(False)
        except Exception:
            pass
        try:
            self.gameEnv.resumeTrainingFor(self.profileName)
        except Exception:
            pass
        try:
            controller = getattr(self.master, 'controller', None)
            if controller and hasattr(controller, 'frames') and 'BotTrainingFrame' in controller.frames:
                controller.frames['BotTrainingFrame'].statusHint.configure(text="")
        except Exception:
            pass
        self.destroy()


class VisualizationFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller
        self._visualizationStrategy = DefaultVisualizationStrategy()
        self._visualizationService = VisualizationService(self.controller.gameEnv.repository)
        self.canvasAgg: Any = None

        ttk.Label(self, text="Visualizations", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)
        self.canvas = tk.Canvas(self, height=600, width=1000)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollableFrame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollableFrame, anchor="nw")
        self.scrollableFrame.bind("<Configure>", self.onFrameConfigure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        def _wheelScroll(event: Any) -> str:
            units = -1 if getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4 else 1
            try:
                self.canvas.yview_scroll(units, 'units')
            except Exception:
                pass
            return "break"
        self.canvas.bind('<MouseWheel>', _wheelScroll)
        self.canvas.bind('<Button-4>', _wheelScroll)
        self.canvas.bind('<Button-5>', _wheelScroll)

        ttk.Label(self.scrollableFrame, text="Select Profile:").pack()
        self.profileSelect = ttk.Combobox(self.scrollableFrame)
        self.profileSelect.pack()

        ttk.Button(self.scrollableFrame, text="Load Profile", command=self.loadProfile).pack(pady=10)
        self.heatmapFrame = tk.Frame(self.scrollableFrame)
        self.heatmapFrame.pack(pady=10)
        ttk.Label(self.heatmapFrame, text="Latest Maze").grid(row=0, column=0, pady=10)
        self.heatmapCanvasLatest = tk.Canvas(self.heatmapFrame, width=300, height=300, bg="white")
        self.heatmapCanvasLatest.grid(row=1, column=0, padx=5)
        ttk.Label(self.heatmapFrame, text="Highest Reward Maze").grid(row=0, column=1, pady=10)
        self.heatmapCanvasHighest = tk.Canvas(self.heatmapFrame, width=300, height=300, bg="white")
        self.heatmapCanvasHighest.grid(row=1, column=1, padx=5)
        ttk.Label(self.heatmapFrame, text="Lowest Reward Maze").grid(row=0, column=2, pady=10)
        self.heatmapCanvasLowest = tk.Canvas(self.heatmapFrame, width=300, height=300, bg="white")
        self.heatmapCanvasLowest.grid(row=1, column=2, padx=5)
        ttk.Label(self.scrollableFrame, text="Reward Graph Visualization:").pack(pady=10)
        self.rewardCanvas = tk.Canvas(self.scrollableFrame, width=800, height=400)
        self.rewardCanvas.pack(pady=10)
        ttk.Label(self.scrollableFrame, text="Q-Table Visualization:").pack(pady=10)
        self.qtableOutput = tk.Text(self.scrollableFrame, height=10, width=50)
        self.qtableOutput.pack(pady=10)
        self.qtableScrollbar = ttk.Scrollbar(self.scrollableFrame, command=self.qtableOutput.yview)
        self.qtableScrollbar.pack(side="right", fill="y")
        self.qtableOutput.config(yscrollcommand=self.qtableScrollbar.set)
        ttk.Label(self.scrollableFrame, text="Statistics:").pack(pady=10)
        self.statisticsOutput = tk.Text(self.scrollableFrame, height=5, width=50)
        self.statisticsOutput.pack(pady=10)
        self.loadProfiles()

        def _reloadProfiles(**_: Any) -> None:
            self.loadProfiles()
        controller.eventBus.subscribe(PROFILE_SAVED, _reloadProfiles)
        controller.eventBus.subscribe(PROFILE_DELETED, _reloadProfiles)

    def onFrameConfigure(self, event: Any) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def loadProfiles(self) -> None:
        profiles = self.controller.gameEnv.profileManager.listProfiles()
        self.profileSelect['values'] = profiles

    def loadProfile(self) -> None:
        selectedProfile = self.profileSelect.get()
        if not selectedProfile:
            messagebox.showerror("Error", "No profile selected.")
            return
        profile = self.controller.gameEnv.profileManager.loadProfile(selectedProfile)
        profileIndex = self.controller.gameEnv.applyProfile(profile)
        bot = self.controller.gameEnv.bots[profileIndex]
        snapshot = self._visualizationService.build_snapshot(selectedProfile, bot)
        self._visualizationStrategy.visualize(self, snapshot)

    def renderVisualizationSnapshot(self, snapshot: VisualizationSnapshot) -> None:
        self._renderHeatmapSnapshot(self.heatmapCanvasLatest, snapshot.latest)
        self._renderHeatmapSnapshot(self.heatmapCanvasHighest, snapshot.highest)
        self._renderHeatmapSnapshot(self.heatmapCanvasLowest, snapshot.lowest)
        self.displayQtableSnapshot(snapshot)
        self.displayStatisticsSnapshot(snapshot)
        self.displayRewardGraphSnapshot(snapshot)

    def _renderHeatmapSnapshot(self, canvas: Any, snapshot: Any) -> None:
        if snapshot is None:
            canvas.delete("all")
            return
        self.displayHeatmap(
            canvas,
            snapshot.maze,
            snapshot.start,
            snapshot.end,
            snapshot.heatmap_data,
        )

    def displayHeatmap(self, canvas: Any, maze: Any, start: Any, end: Any, heatmapData: Any) -> None:
        try:
            DisplayTools.displayHeatmap(canvas, maze, start, end, heatmapData)
        except Exception:
            pass

    def displayQtable(self, bot: Any, profileIndex: int) -> None:
        snapshot = self._visualizationService.build_snapshot(bot.profileName, bot)
        self.displayQtableSnapshot(snapshot)

    def displayQtableSnapshot(self, snapshot: VisualizationSnapshot) -> None:
        self.qtableOutput.delete("1.0", tk.END)
        if snapshot.q_table is not None:
            topValues = self.getTopQValues(snapshot.q_table)
            actionSpecs = list(snapshot.action_specs)
            self.qtableOutput.insert(tk.END, "Top Q-Table Values:\n")
            for i, (qValue, (state, actions)) in enumerate(topValues):
                bestActionIndex = int(np.argmax(actions))
                bestAction = self.getActionLabel(bestActionIndex, actionSpecs)
                bestQValue = qValue
                rankedActions = self.formatRankedActions(actions, actionSpecs)
                self.qtableOutput.insert(tk.END, f"Rank {i+1}:\n")
                self.qtableOutput.insert(tk.END, f"  State: {self.formatStateSummary(state)}\n")
                self.qtableOutput.insert(tk.END, f"  Best Action: {bestAction}\n")
                self.qtableOutput.insert(tk.END, f"  Best Q-value: {bestQValue}\n\n")
                if rankedActions:
                    self.qtableOutput.insert(tk.END, f"  Actions: {rankedActions}\n\n")
        else:
            self.qtableOutput.insert(tk.END, "No tabular Q-table available for this bot.\n")

    def displayStatistics(self, bot: Any, profileIndex: int) -> None:
        snapshot = self._visualizationService.build_snapshot(bot.profileName, bot)
        self.displayStatisticsSnapshot(snapshot)

    def displayStatisticsSnapshot(self, snapshot: VisualizationSnapshot) -> None:
        self.statisticsOutput.delete("1.0", tk.END)
        profileData = dict(snapshot.profile_stats or {})
        self.statisticsOutput.insert(tk.END, f"Total Steps: {profileData.get('total_steps', 0)}\n")
        self.statisticsOutput.insert(tk.END, f"Non-Repeating Steps: {profileData.get('non_repeating_steps_taken', 0)}\n")
        self.statisticsOutput.insert(tk.END, f"Times Revisited Squares: {profileData.get('times_revisited_squares', 0)}\n")
        self.statisticsOutput.insert(tk.END, f"Times Bot Hit Wall: {profileData.get('times_hit_wall', 0)}\n")
        self.statisticsOutput.insert(tk.END, f"Times Bot Hit Enemy: {profileData.get('times_hit_enemy', 0)}\n")

    def displayRewardGraph(self, bot: Any) -> None:
        snapshot = self._visualizationService.build_snapshot(bot.profileName, bot)
        self.displayRewardGraphSnapshot(snapshot)

    def displayRewardGraphSnapshot(self, snapshot: VisualizationSnapshot) -> None:
        if self.canvasAgg:
            self.canvasAgg.get_tk_widget().destroy()
        rewardFilenames = [snapshot.reward_path]
        grapher = RewardGrapher(rewardFilenames)
        self.canvasAgg = grapher.run(self.rewardCanvas)

    def getTopQValues(self, qTable: dict[Any, Any], n: int = 10) -> list[Any]:
        qTableItems = list(qTable.items())
        topItems = []
        for item in qTableItems:
            qValue = np.max(item[1])
            if len(topItems) < n:
                topItems.append((qValue, item))
                topItems.sort(reverse=True, key=lambda item: item[0])
            else:
                if qValue > topItems[-1][0]:
                    topItems[-1] = (qValue, item)
                    topItems.sort(reverse=True, key=lambda item: item[0])
        return topItems

    @staticmethod
    def getActionLabel(actionIndex: int, actionSpecs: list[Any] | None = None) -> str:
        if actionSpecs and 0 <= int(actionIndex) < len(actionSpecs):
            spec = actionSpecs[int(actionIndex)]
            name = str(getattr(spec, "name", f"action_{actionIndex}"))
            kind = str(getattr(spec, "kind", "action"))
            return f"{VisualizationFrame.humanizeActionName(name)} [{kind}]"
        return f"Action {int(actionIndex)}"

    @staticmethod
    def humanizeActionName(name: str) -> str:
        return name.replace("_", " ").strip().title()

    @staticmethod
    def formatRankedActions(actions: Any, actionSpecs: list[Any]) -> str:
        try:
            ranked = sorted(
                enumerate(np.asarray(actions).tolist()),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        except Exception:
            return ""
        return ", ".join(
            f"{VisualizationFrame.getActionLabel(index, actionSpecs)}={float(value):.3f}"
            for index, value in ranked
        )

    @staticmethod
    def formatStateSummary(state: Any) -> str:
        if not isinstance(state, tuple):
            return str(state)
        if len(state) >= 3:
            parts = [
                f"position={state[0]}",
                f"walls={state[1]}",
                f"goal={state[2]}",
            ]
            if len(state) > 3 and state[3]:
                parts.append(f"entities={state[3]}")
            return ", ".join(parts)
        return str(state)
