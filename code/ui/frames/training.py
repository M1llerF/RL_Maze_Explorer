import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any


class BotTrainingFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller

        self.visualizationWindow = None
        self.trainingActive = False
        self.lastSelectedProfile = ""
        self._logInterval = 1
        self._logLastRound = 0

        ttk.Label(self, text="Bot Training", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)

        ttk.Label(self, text="Select Profile:").pack()
        self.profileSelect = ttk.Combobox(self, state="readonly")
        self.profileSelect.pack()

        ttk.Label(self, text="Number of Rounds:").pack()
        self.roundsEntry = ttk.Entry(self)
        self.roundsEntry.pack()

        ttk.Label(self, text="Maze Source:").pack(pady=(10, 0))
        self.mazeMode = ttk.Combobox(self, values=["Random", "Fixed (Builder)", "Pool"], state="readonly")
        self.mazeMode.set("Random")
        self.mazeMode.pack()

        def onMazeModeChange(event: Any = None) -> None:
            mode = self.mazeMode.get()
            if mode == "Fixed (Builder)":
                if not self.controller.gameEnv.fixedMazeActive or self.controller.gameEnv.fixedMazeState is None:
                    path = filedialog.askopenfilename(
                        title="Choose a maze JSON to use as Fixed",
                        initialdir='mazes',
                        filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
                    )
                    if path:
                        try:
                            with open(path, 'r') as f:
                                state = json.load(f)
                            self.controller.gameEnv.setFixedMaze(state)
                            messagebox.showinfo("Fixed Maze Set", f"Using {os.path.basename(path)} for training.")
                        except Exception as e:
                            messagebox.showerror("Invalid Maze", f"Could not load maze: {e}")
                            self.mazeMode.set("Random")
                    else:
                        self.mazeMode.set("Random")

        self.mazeMode.bind("<<ComboboxSelected>>", onMazeModeChange)

        self.openBuilderBtn = ttk.Button(self, text="Open Maze Builder", command=lambda: self.controller.showMazeBuilder())
        self.openBuilderBtn.pack(pady=6)

        actions = ttk.Frame(self)
        actions.pack(pady=10)
        self.startBtn = ttk.Button(actions, text="Start Training", command=self.startTraining)
        self.startBtn.pack(side=tk.LEFT, padx=6)
        self.stopBtn = ttk.Button(actions, text="Stop", command=self.stopTraining, state="disabled")
        self.stopBtn.pack(side=tk.LEFT, padx=6)
        self.trainingProgress = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self.trainingProgress.pack(pady=10)
        self.logOutput = tk.Text(self, height=10, width=50)
        self.logOutput.pack(pady=10)

        ttk.Button(self, text="Open Visualization", command=self.openVisualization).pack(pady=10)
        self.statusHint = tk.Label(self, text="", fg="#805b00")
        self.statusHint.pack(pady=(0, 6))

        self.loadProfiles()

    def onShow(self) -> None:
        self.loadProfiles()

    def loadProfiles(self) -> None:
        profiles = self.controller.gameEnv.profileManager.listProfiles()
        self.profileSelect['values'] = profiles
        if self.lastSelectedProfile and self.lastSelectedProfile in profiles:
            try:
                self.profileSelect.set(self.lastSelectedProfile)
            except Exception:
                pass

    def startTraining(self) -> None:
        if self.trainingActive:
            return
        selectedProfile = self.profileSelect.get()
        if not selectedProfile:
            messagebox.showerror("Error", "No profile selected.")
            return

        roundsTxt = self.roundsEntry.get()
        if not roundsTxt.isdigit():
            messagebox.showerror("Error", "Number of rounds must be a positive integer.")
            return
        rounds = int(roundsTxt)

        mode = self.mazeMode.get() or "Random"
        if mode == "Fixed (Builder)":
            if not self.controller.gameEnv.fixedMazeActive or self.controller.gameEnv.fixedMazeState is None:
                messagebox.showerror("No Fixed Maze", "No fixed maze set. Open Maze Builder to create or load one, then click 'Use In Training'.")
                return

        self.trainingProgress['maximum'] = rounds
        self.trainingProgress['value'] = 0
        self._logInterval = max(1, rounds // 100)
        self._logLastRound = 0
        self.logOutput.delete("1.0", tk.END)
        self.logOutput.insert(tk.END, f"Training started for {selectedProfile} with {rounds} rounds...\n")
        self.lastSelectedProfile = selectedProfile
        self.setControlsEnabled(False)
        self.trainingActive = True
        try:
            self.stopBtn.configure(state="normal")
        except Exception:
            pass

        def onProgress(done: int, total: int):
            try:
                self.controller.root.after(0, self.updateProgress, done, total)
            except Exception:
                pass

        def onError(err: Exception) -> None:
            def _report():
                try:
                    self.logOutput.insert(tk.END, f"Training error: {err}\n")
                    self.logOutput.see(tk.END)
                except Exception:
                    pass
                self.trainingActive = False
                self.setControlsEnabled(True)
                try:
                    self.stopBtn.configure(state="disabled")
                except Exception:
                    pass
            try:
                self.controller.root.after(0, _report)
            except Exception:
                pass

        def onComplete() -> None:
            def _done():
                self.trainingActive = False
                self.setControlsEnabled(True)
                try:
                    self.stopBtn.configure(state="disabled")
                except Exception:
                    pass
            try:
                self.controller.root.after(0, _done)
            except Exception:
                pass

        poolSize = max(5, min(25, rounds // 10 or 5)) if mode == "Pool" else 0
        self.controller.trainingController.start(
            profileName=selectedProfile,
            rounds=rounds,
            mazeMode=mode,
            poolSize=poolSize or 20,
            onProgress=onProgress,
            onError=onError,
            onComplete=onComplete,
        )

    def setControlsEnabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        entryState = "normal" if enabled else "disabled"
        try:
            self.profileSelect.configure(state=state)
        except Exception:
            pass
        try:
            self.mazeMode.configure(state=state)
        except Exception:
            pass
        try:
            self.roundsEntry.configure(state=entryState)
        except Exception:
            pass
        try:
            self.openBuilderBtn.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            pass

    def stopTraining(self) -> None:
        if not self.trainingActive:
            return
        try:
            self.stopBtn.configure(state="disabled")
        except Exception:
            pass
        try:
            self.controller.trainingController.stop()
        except Exception:
            pass

    def updateProgress(self, completedRounds: int, totalRounds: int) -> None:
        self.trainingProgress['value'] = completedRounds
        if (
            completedRounds == totalRounds
            or completedRounds == 1
            or completedRounds - self._logLastRound >= self._logInterval
        ):
            self.logOutput.insert(tk.END, f"Completed round {completedRounds}/{totalRounds}\n")
            self.logOutput.see(tk.END)
            self._logLastRound = completedRounds
        if completedRounds == totalRounds:
            self.logOutput.insert(tk.END, "Training completed.\n")
            self.logOutput.see(tk.END)
            self.trainingActive = False
            self.setControlsEnabled(True)

    def cancelTrainingPoll(self) -> None:
        return

    def openVisualization(self) -> None:
        from ui.frames.visualization import VisualizationWindow

        selectedProfile = self.profileSelect.get()
        if not selectedProfile:
            messagebox.showerror("Error", "No profile selected.")
            return
        profile = self.controller.gameEnv.profileManager.loadProfile(selectedProfile)
        profileIndex = self.controller.gameEnv.applyProfile(profile)

        existingWindow = self.visualizationWindow
        if existingWindow is not None and existingWindow.winfo_exists():
            existingWindow.focus()
        else:
            self.visualizationWindow = VisualizationWindow(self.controller.root, self.controller.gameEnv, selectedProfile, profileIndex)
            try:
                self.logOutput.insert(tk.END, "Opened visualization. Training is PAUSED while the window is open.\n")
                self.logOutput.see(tk.END)
                self.statusHint.configure(text="Note: Training is paused while visualization is open.")
            except Exception:
                pass

