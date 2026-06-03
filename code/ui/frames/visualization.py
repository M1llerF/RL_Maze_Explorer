# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np

from displayTools import DisplayTools
from rewardGrapher import RewardGrapher
from visualizationStrategy import DefaultVisualizationStrategy


class VisualizationWindow(tk.Toplevel):
    def __init__(self, parent: Any, gameEnv: Any, profileName: str, profileIndex: int) -> None:
        super().__init__(parent)
        self.gameEnv = gameEnv
        self.profileName = profileName
        self.profileIndex = profileIndex
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
        self.stepsVar = tk.IntVar(value=20)
        self.stepsScale = tk.Scale(controls, from_=1, to=200, orient=tk.HORIZONTAL, variable=self.stepsVar, length=200)
        self.stepsScale.pack(side=tk.LEFT, padx=5)
        self.paused = False
        self.pauseBtn = ttk.Button(controls, text="Pause", command=self.togglePause)
        self.pauseBtn.pack(side=tk.LEFT, padx=5)

        self.afterId = None
        self.visualize = True
        self.protocol("WM_DELETE_WINDOW", self.onClose)

        try:
            self.gameEnv.pauseTrainingFor(self.profileName)
        except Exception:
            pass

        self.updateVisualization()

    def updateVisualization(self) -> None:
        if not self.visualize:
            return
        self.canvas.delete("all")
        bot = self.gameEnv.bots[self.profileIndex]
        if not self.paused:
            try:
                finished = bot.stepVisualization(maxSteps=self.stepsVar.get())
                if finished:
                    self.gameEnv.resetEnvironment(self.profileIndex)
            except AttributeError:
                pass
        botPosition = bot.position
        self.displayWithBotAndHeatmap(botPosition, bot.statistics.getVisitedPositions())
        self.afterId = self.after(100, self.updateVisualization)

    def togglePause(self) -> None:
        self.paused = not self.paused
        self.pauseBtn.configure(text="Resume" if self.paused else "Pause")

    def displayWithBot(self, botPosition: tuple[int, int]) -> None:
        maze = self.gameEnv.maze
        cellWidth = self.canvas.winfo_width() / maze.width
        cellHeight = self.canvas.winfo_height() / maze.height
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.grid[y][x] == 1:
                    self.canvas.create_rectangle(x * cellWidth, y * cellHeight,
                                                 (x + 1) * cellWidth, (y + 1) * cellHeight,
                                                 fill="black")
        start = maze.getStart()
        end = maze.end
        self.canvas.create_rectangle(start[1] * cellWidth, start[0] * cellHeight,
                                     (start[1] + 1) * cellWidth, (start[0] + 1) * cellHeight,
                                     fill="blue")
        self.canvas.create_rectangle(end[1] * cellWidth, end[0] * cellHeight,
                                     (end[1] + 1) * cellWidth, (end[0] + 1) * cellHeight,
                                     fill="green")
        self.canvas.create_oval(botPosition[1] * cellWidth, botPosition[0] * cellHeight,
                                (botPosition[1] + 1) * cellWidth, (botPosition[0] + 1) * cellHeight,
                                fill="red")

    def displayWithBotAndHeatmap(self, botPosition: tuple[int, int], visitedPositions: dict[tuple[int, int], int]) -> None:
        maze = self.gameEnv.maze
        cellWidth = self.canvas.winfo_width() / maze.width
        cellHeight = self.canvas.winfo_height() / maze.height
        heatmap = np.zeros((maze.height, maze.width))
        for (x, y), count in visitedPositions.items():
            heatmap[x, y] = count
        maxHeat = heatmap.max() if heatmap.max() > 0 else 1
        cmap = plt.get_cmap("Reds")
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.grid[y][x] == 1:
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
        start = maze.getStart()
        end = maze.end
        self.canvas.create_rectangle(start[1] * cellWidth, start[0] * cellHeight,
                                     (start[1] + 1) * cellWidth, (start[0] + 1) * cellHeight,
                                     fill="blue")
        self.canvas.create_rectangle(end[1] * cellWidth, end[0] * cellHeight,
                                     (end[1] + 1) * cellWidth, (end[0] + 1) * cellHeight,
                                     fill="green")
        self.canvas.create_oval(botPosition[1] * cellWidth, botPosition[0] * cellHeight,
                                (botPosition[1] + 1) * cellWidth, (botPosition[0] + 1) * cellHeight,
                                fill="red")

    def onClose(self) -> None:
        self.visualize = False
        if self.afterId is not None:
            self.after_cancel(self.afterId)
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
        self._visualizationStrategy.visualize(self, bot, profileIndex)

    def displayHeatmap(self, canvas: Any, maze: Any, start: Any, end: Any, heatmapData: Any) -> None:
        try:
            DisplayTools.displayHeatmap(canvas, maze, start, end, heatmapData)
        except Exception:
            pass

    def displayQtable(self, bot: Any, profileIndex: int) -> None:
        self.qtableOutput.delete("1.0", tk.END)
        if hasattr(bot, 'qLearning') and hasattr(bot.qLearning, 'qTable'):
            topValues = self.getTopQValues(bot, profileIndex)
            self.qtableOutput.insert(tk.END, "Top Q-Table Values:\n")
            for i, (qValue, (state, actions)) in enumerate(topValues):
                position, surrounding, stepCount = state
                bestActionIndex = int(np.argmax(actions))
                bestAction = self.getActionLabel(bestActionIndex)
                bestQValue = qValue
                self.qtableOutput.insert(tk.END, f"Rank {i+1}:\n")
                self.qtableOutput.insert(tk.END, f"  Current Position: {position}\n")
                self.qtableOutput.insert(tk.END, f"  Surrounding: {surrounding}\n")
                self.qtableOutput.insert(tk.END, f"  Step Count: {stepCount}\n")
                self.qtableOutput.insert(tk.END, f"  Best Action: {bestAction}\n")
                self.qtableOutput.insert(tk.END, f"  Best Q-value: {bestQValue}\n\n")
        else:
            self.qtableOutput.insert(tk.END, "No tabular Q-table available for this bot.\n")

    def displayStatistics(self, bot: Any, profileIndex: int) -> None:
        self.statisticsOutput.delete("1.0", tk.END)
        try:
            repo = self.controller.gameEnv.repository
            profileData = repo.readProfileStats(bot.profileName) or {}
        except Exception:
            profileData = {}
        self.statisticsOutput.insert(tk.END, f"Total Steps: {profileData.get('total_steps', 0)}\n")
        self.statisticsOutput.insert(tk.END, f"Non-Repeating Steps: {profileData.get('non_repeating_steps_taken', 0)}\n")
        self.statisticsOutput.insert(tk.END, f"Times Revisited Squares: {profileData.get('times_revisited_squares', 0)}\n")
        self.statisticsOutput.insert(tk.END, f"Times Bot Hit Wall: {profileData.get('times_hit_wall', 0)}\n")

    def displayRewardGraph(self, bot: Any) -> None:
        if self.canvasAgg:
            self.canvasAgg.get_tk_widget().destroy()
        try:
            rewardPath = self.controller.gameEnv.repository.rewardsPath(bot.profileName)
        except Exception:
            rewardPath = f'profiles/{bot.profileName}/SimulationRewards.txt'
        rewardFilenames = [rewardPath]
        grapher = RewardGrapher(rewardFilenames)
        self.canvasAgg = grapher.run(self.rewardCanvas)

    def getTopQValues(self, bot: Any, profileIndex: int, n: int = 10) -> list[Any]:
        qTable = bot.qLearning.qTable
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

    def getActionLabel(self, actionIndex: int) -> str:
        actionLabels = ["Up", "Down", "Left", "Right"]
        return actionLabels[actionIndex]
