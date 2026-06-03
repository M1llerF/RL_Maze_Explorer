import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any, Optional

from bots.dqnlearning.config import DQNConfig
from ui.scrollable import VerticalScrolledFrame


class BotTrainingFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller

        self.visualizationWindow = None
        self.trainingActive = False
        self.lastSelectedProfile = ""
        self._logInterval = 1
        self._logLastRound = 0
        self._stepPollAfterId: str | None = None
        self._lastWarmupState: bool | None = None
        self._lastEpsilonValue: float | None = None
        self._lastMazeSize: tuple[int, int] | None = None
        self._loadedWarmupStore: Optional[Any] = None
        self._loadedWarmupSourceProfile: str | None = None
        self._warmupOptionsByLabel: dict[str, Any] = {}
        self._collectPollAfterId: str | None = None

        scrollHost = VerticalScrolledFrame(self)
        scrollHost.pack(fill=tk.BOTH, expand=True)
        content = scrollHost.content

        ttk.Label(content, text="Bot Training", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)

        ttk.Label(content, text="Select Profile:").pack()
        self.profileSelect = ttk.Combobox(content, state="readonly")
        self.profileSelect.pack()

        ttk.Label(content, text="Number of Rounds:").pack()
        self.roundsEntry = ttk.Entry(content)
        self.roundsEntry.pack()

        ttk.Label(content, text="Maze Source:").pack(pady=(10, 0))
        self.mazeMode = ttk.Combobox(content, values=["Random", "Fixed (Builder)", "Pool"], state="readonly")
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
        self.profileSelect.bind("<<ComboboxSelected>>", lambda _e: self._updateWarmupStatus())

        self.openBuilderBtn = ttk.Button(content, text="Open Maze Builder", command=lambda: self.controller.showMazeBuilder())
        self.openBuilderBtn.pack(pady=6)

        # ── Warmup store ────────────────────────────────────────────────
        warmupFrame = ttk.LabelFrame(content, text="Warmup Store", padding=(8, 4))
        warmupFrame.pack(pady=(6, 0), fill=tk.X, padx=20)

        self.warmupStatusLabel = ttk.Label(warmupFrame, text="No warmup store")
        self.warmupStatusLabel.pack()

        collectModeRow = ttk.Frame(warmupFrame)
        collectModeRow.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(collectModeRow, text="Collect by:").pack(side=tk.LEFT)
        self.warmupCollectModeVar = tk.StringVar(value="transitions")
        ttk.Radiobutton(
            collectModeRow, text="Transitions", variable=self.warmupCollectModeVar,
            value="transitions", command=self._onWarmupCollectModeChanged,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(
            collectModeRow, text="Completions", variable=self.warmupCollectModeVar,
            value="completions", command=self._onWarmupCollectModeChanged,
        ).pack(side=tk.LEFT, padx=(4, 0))

        warmupAmountRow = ttk.Frame(warmupFrame)
        warmupAmountRow.pack(fill=tk.X, padx=4, pady=(2, 0))
        self.warmupAmountLabel = ttk.Label(warmupAmountRow, text="Transitions:")
        self.warmupAmountLabel.pack(side=tk.LEFT)
        self.warmupTransitionsVar = tk.StringVar(value=str(DQNConfig.replayWarmupSteps))
        self.warmupTransitionsEntry = ttk.Entry(warmupAmountRow, textvariable=self.warmupTransitionsVar, width=10)
        self.warmupTransitionsEntry.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(warmupFrame, text="Warmup Source:").pack(pady=(4, 0))
        self.warmupSourceSelect = ttk.Combobox(warmupFrame, state="disabled")
        self.warmupSourceSelect.pack(fill=tk.X, padx=4)
        self.warmupSourceSelect.bind("<<ComboboxSelected>>", lambda _e: self._onWarmupSourceChanged())

        warmupBtns = ttk.Frame(warmupFrame)
        warmupBtns.pack(pady=(4, 0))
        self.collectWarmupBtn = ttk.Button(warmupBtns, text="Collect New", command=self._collectWarmup)
        self.collectWarmupBtn.pack(side=tk.LEFT, padx=4)
        self.clearWarmupBtn = ttk.Button(warmupBtns, text="Clear", command=self._clearWarmup)
        self.clearWarmupBtn.pack(side=tk.LEFT, padx=4)

        self.useWarmupVar = tk.BooleanVar(value=False)
        self.useWarmupCheck = ttk.Checkbutton(
            warmupFrame,
            text="Inject warmup at training start",
            variable=self.useWarmupVar,
        )
        self.useWarmupCheck.pack(pady=(4, 0))
        self.useWarmupCheck.state(["disabled"])
        # ────────────────────────────────────────────────────────────────

        actions = ttk.Frame(content)
        actions.pack(pady=10)
        self.startBtn = ttk.Button(actions, text="Start Training", command=self.startTraining)
        self.startBtn.pack(side=tk.LEFT, padx=6)
        self.stopBtn = ttk.Button(actions, text="Stop", command=self.stopTraining, state="disabled")
        self.stopBtn.pack(side=tk.LEFT, padx=6)
        self.trainingProgress = ttk.Progressbar(content, orient="horizontal", length=200, mode="determinate")
        self.trainingProgress.pack(pady=10)
        self.stepCounterLabel = ttk.Label(content, text="Episode Steps: 0")
        self.stepCounterLabel.pack()
        self.epsilonLabel = ttk.Label(content, text="Epsilon: n/a")
        self.epsilonLabel.pack()
        epsilonControls = ttk.Frame(content)
        epsilonControls.pack(pady=(4, 0))
        ttk.Label(epsilonControls, text="Manual Epsilon (0-1):").pack(side=tk.LEFT, padx=(0, 6))
        self.epsilonEntry = ttk.Entry(epsilonControls, width=8)
        self.epsilonEntry.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(epsilonControls, text="Apply", command=self.applyManualEpsilon).pack(side=tk.LEFT, padx=3)
        ttk.Button(epsilonControls, text="Clear Override", command=self.clearManualEpsilon).pack(side=tk.LEFT, padx=3)
        logFrame = ttk.Frame(content)
        logFrame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.logOutput = tk.Text(logFrame, height=10, width=50)
        logScrollbar = ttk.Scrollbar(logFrame, orient="vertical", command=self.logOutput.yview)
        self.logOutput.configure(yscrollcommand=logScrollbar.set)
        self.logOutput.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        logScrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(content, text="Open Visualization", command=self.openVisualization).pack(pady=10)
        self.statusHint = tk.Label(content, text="", fg="#805b00")
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
        self._updateWarmupStatus()

    # ── Warmup collection poll ──────────────────────────────────────────────

    def _startCollectPoll(self) -> None:
        self._stopCollectPoll()
        self._pollCollectProgress()

    def _stopCollectPoll(self) -> None:
        if self._collectPollAfterId is not None:
            try:
                self.after_cancel(self._collectPollAfterId)
            except Exception:
                pass
            self._collectPollAfterId = None

    def _pollCollectProgress(self) -> None:
        if not self.controller.trainingController.isCollecting():
            self._collectPollAfterId = None
            return
        try:
            profile = self.profileSelect.get() or self.lastSelectedProfile
            bot = next(
                (b for b in self.controller.gameEnv.bots if getattr(b, "profileName", "") == profile),
                None,
            )
            if bot is not None:
                replay = getattr(getattr(bot, "agent", None), "replay", None)
                current = len(replay) if replay is not None else 0
                target = int(getattr(getattr(bot, "config", None), "replayWarmupSteps", 2000))
                self.warmupStatusLabel.configure(text=f"Collecting… {current}/{target} transitions")
        except Exception:
            pass
        self._collectPollAfterId = self.after(150, self._pollCollectProgress)

    # ── Warmup helpers ───────────────────────────────────────────────────────

    def _updateWarmupStatus(self) -> None:
        """Refresh the warmup status label and checkbox state."""
        from bots.dqnlearning.warmup_store import WarmupStore
        profile = self.profileSelect.get() or self.lastSelectedProfile
        if not profile:
            self.warmupStatusLabel.configure(text="No warmup store")
            self._loadedWarmupStore = None
            self._loadedWarmupSourceProfile = None
            self._warmupOptionsByLabel = {}
            self.warmupSourceSelect.set("")
            self.warmupSourceSelect.configure(state="disabled")
            self.warmupSourceSelect["values"] = ()
            self.useWarmupCheck.state(["disabled"])
            return

        try:
            profileObj = self.controller.gameEnv.profileManager.loadProfile(profile)
        except Exception:
            profileObj = None

        compatibility = None
        if profileObj is not None:
            compatibility = WarmupStore.inferCompatibilityForProfile(profileObj)

        if compatibility is None:
            self.warmupStatusLabel.configure(text="No compatible warmup stores")
            self._loadedWarmupStore = None
            self._loadedWarmupSourceProfile = None
            self._warmupOptionsByLabel = {}
            self.warmupSourceSelect.set("")
            self.warmupSourceSelect.configure(state="disabled")
            self.warmupSourceSelect["values"] = ()
            self.useWarmupCheck.state(["disabled"])
            self.useWarmupVar.set(False)
            return

        fingerprint, inputDim = compatibility
        options = WarmupStore.loadCompatibleOptions(
            self.controller.gameEnv.repository,
            list(self.controller.gameEnv.profileManager.listProfiles()),
            selectedProfile=profile,
            fingerprint=fingerprint,
            inputDim=inputDim,
        )

        if not options:
            self.warmupStatusLabel.configure(text="No compatible warmup stores")
            self._loadedWarmupStore = None
            self._loadedWarmupSourceProfile = None
            self._warmupOptionsByLabel = {}
            self.warmupSourceSelect.set("")
            self.warmupSourceSelect.configure(state="disabled")
            self.warmupSourceSelect["values"] = ()
            self.useWarmupCheck.state(["disabled"])
            self.useWarmupVar.set(False)
            return

        self._warmupOptionsByLabel = {option.label: option for option in options}
        values = [option.label for option in options]
        self.warmupSourceSelect["values"] = values

        selectedLabel = self.warmupSourceSelect.get()
        if selectedLabel not in self._warmupOptionsByLabel:
            selectedLabel = values[0]
            self.warmupSourceSelect.set(selectedLabel)
        self.warmupSourceSelect.configure(state="readonly")
        self._applyWarmupSelection(selectedLabel)
        selectedOption = self._warmupOptionsByLabel[selectedLabel]
        sourceNote = "local" if selectedOption.isLocal else f"from {selectedOption.profileName}"
        self.warmupStatusLabel.configure(
            text=(
                f"Compatible warmups: {len(options)} "
                f"(selected {sourceNote}, {selectedOption.transitionCount} transitions)"
            )
        )
        self.useWarmupCheck.state(["!disabled"])

    def _applyWarmupSelection(self, label: str) -> None:
        option = self._warmupOptionsByLabel.get(label)
        if option is None:
            self._loadedWarmupStore = None
            self._loadedWarmupSourceProfile = None
            return
        self._loadedWarmupStore = option.store
        self._loadedWarmupSourceProfile = option.profileName

    def _onWarmupSourceChanged(self) -> None:
        label = self.warmupSourceSelect.get()
        if not label:
            return
        self._applyWarmupSelection(label)
        option = self._warmupOptionsByLabel.get(label)
        if option is None:
            return
        sourceNote = "local" if option.isLocal else f"from {option.profileName}"
        self.warmupStatusLabel.configure(
            text=(
                f"Compatible warmups: {len(self._warmupOptionsByLabel)} "
                f"(selected {sourceNote}, {option.transitionCount} transitions)"
            )
        )

    def _onWarmupCollectModeChanged(self) -> None:
        collectMode = self.warmupCollectModeVar.get()
        if collectMode == "completions":
            self.warmupAmountLabel.configure(text="Completions:")
            self.warmupTransitionsVar.set("10")
        else:
            self.warmupAmountLabel.configure(text="Transitions:")
            self.warmupTransitionsVar.set(str(DQNConfig.replayWarmupSteps))

    def _collectWarmup(self) -> None:
        profile = self.profileSelect.get()
        if not profile:
            messagebox.showerror("No Profile", "Select a profile before collecting warmup.")
            return
        if self.trainingActive or self.controller.trainingController.isCollecting():
            messagebox.showinfo("Busy", "Wait for the current run to finish.")
            return
        collectMode = self.warmupCollectModeVar.get()
        amount = self._parseWarmupAmount(collectMode)
        if amount is None:
            return

        nTransitions = amount if collectMode == "transitions" else None
        nCompletions = amount if collectMode == "completions" else None
        unitLabel = "transitions" if collectMode == "transitions" else "completions"

        mode = self.mazeMode.get() or "Random"
        self.collectWarmupBtn.configure(state="disabled")
        self.clearWarmupBtn.configure(state="disabled")
        self.startBtn.configure(state="disabled")
        self.logOutput.insert(tk.END, f"Collecting warmup for {profile}…\n")
        self.logOutput.see(tk.END)
        self.warmupStatusLabel.configure(text=f"Collecting… 0 {unitLabel}")

        def onProgress(current: int, total: int) -> None:
            def _update() -> None:
                self.warmupStatusLabel.configure(text=f"Collecting… {current}/{total} {unitLabel}")
            self.controller.root.after(0, _update)

        def onComplete(store: Any) -> None:
            def _finish() -> None:
                self._stopCollectPoll()
                self.logOutput.insert(
                    tk.END,
                    f"Warmup collected: {store.transitionCount} transitions — saved.\n",
                )
                self.logOutput.see(tk.END)
                self.collectWarmupBtn.configure(state="normal")
                self.clearWarmupBtn.configure(state="normal")
                self.startBtn.configure(state="normal")
                self._updateWarmupStatus()
            self.controller.root.after(0, _finish)

        def onError(exc: Exception) -> None:
            def _report() -> None:
                self._stopCollectPoll()
                self.logOutput.insert(tk.END, f"Warmup collection failed: {exc}\n")
                self.logOutput.see(tk.END)
                self.collectWarmupBtn.configure(state="normal")
                self.clearWarmupBtn.configure(state="normal")
                self.startBtn.configure(state="normal")
                self._updateWarmupStatus()
            self.controller.root.after(0, _report)

        self.controller.trainingController.collectWarmup(
            profileName=profile,
            mazeMode=mode,
            nTransitions=nTransitions,
            nCompletions=nCompletions,
            onProgress=onProgress,
            onComplete=onComplete,
            onError=onError,
        )

    def _clearWarmup(self) -> None:
        from bots.dqnlearning.warmup_store import WarmupStore
        sourceProfile = self._loadedWarmupSourceProfile or self.profileSelect.get() or self.lastSelectedProfile
        if not sourceProfile:
            return
        store = WarmupStore.load(self.controller.gameEnv.repository, sourceProfile)
        if store is None:
            self._updateWarmupStatus()
            return
        if messagebox.askyesno("Clear Warmup", f"Delete the warmup store for '{sourceProfile}'?"):
            store.clear(self.controller.gameEnv.repository, sourceProfile)
            self._loadedWarmupStore = None
            self._loadedWarmupSourceProfile = None
            self.useWarmupVar.set(False)
            self._updateWarmupStatus()
            self.logOutput.insert(tk.END, "Warmup store cleared.\n")
            self.logOutput.see(tk.END)

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
        self._lastWarmupState = None
        self._lastEpsilonValue = None
        self.stepCounterLabel.configure(text="Episode Steps: 0")
        self.epsilonLabel.configure(text="Epsilon: n/a")
        self.setControlsEnabled(False)
        self.trainingActive = True
        self._lastMazeSize = self.controller.gameEnv.getMazeSize()
        self._startStepPoll()
        self.stopBtn.configure(state="normal")

        def onProgress(done: int, total: int):
            self.controller.root.after(0, self.updateProgress, done, total)

        def onError(err: Exception) -> None:
            def _report():
                self.logOutput.insert(tk.END, f"Training error: {err}\n")
                self.logOutput.see(tk.END)
                self.trainingActive = False
                self.setControlsEnabled(True)
                self._stopStepPoll()
                self._lastWarmupState = None
                self._lastEpsilonValue = None
                self._lastMazeSize = None
                self.stepCounterLabel.configure(text="Episode Steps: 0")
                self.epsilonLabel.configure(text="Epsilon: n/a")
                self.stopBtn.configure(state="disabled")
            self.controller.root.after(0, _report)

        def onComplete() -> None:
            def _done():
                self.trainingActive = False
                self.setControlsEnabled(True)
                self._stopStepPoll()
                self._lastWarmupState = None
                self._lastEpsilonValue = None
                self._lastMazeSize = None
                self.stepCounterLabel.configure(text="Episode Steps: 0")
                self.epsilonLabel.configure(text="Epsilon: n/a")
                self.stopBtn.configure(state="disabled")
            self.controller.root.after(0, _done)

        poolSize = max(5, min(25, rounds // 10 or 5)) if mode == "Pool" else 0
        warmupStore = self._loadedWarmupStore if self.useWarmupVar.get() else None
        warmupTransitionCount = self._parseWarmupAmount("transitions") if warmupStore is not None else None
        if warmupStore is not None and warmupTransitionCount is None:
            return
        self.controller.trainingController.start(
            profileName=selectedProfile,
            rounds=rounds,
            mazeMode=mode,
            poolSize=poolSize or 20,
            warmupStore=warmupStore,
            warmupTransitionCount=warmupTransitionCount,
            onProgress=onProgress,
            onError=onError,
            onComplete=onComplete,
        )

    def _parseWarmupAmount(self, mode: str) -> int | None:
        label = "transitions" if mode == "transitions" else "completions"
        raw = self.warmupTransitionsVar.get().strip()
        if not raw:
            messagebox.showerror("Invalid Warmup", f"Warmup {label} must be a positive integer.")
            return None
        try:
            value = int(raw)
        except Exception:
            messagebox.showerror("Invalid Warmup", f"Warmup {label} must be a positive integer.")
            return None
        if value <= 0:
            messagebox.showerror("Invalid Warmup", f"Warmup {label} must be greater than zero.")
            return None
        return value

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
        self._stopStepPoll()
        self._lastWarmupState = None
        self._lastEpsilonValue = None
        self._lastMazeSize = None
        self.stepCounterLabel.configure(text="Episode Steps: 0")
        self.epsilonLabel.configure(text="Epsilon: n/a")

    def updateProgress(self, completedRounds: int, totalRounds: int) -> None:
        self.trainingProgress['value'] = completedRounds
        if (
            completedRounds == totalRounds
            or completedRounds == 1
            or completedRounds - self._logLastRound >= self._logInterval
        ):
            profile = self.controller.trainingController.activeProfile or self.lastSelectedProfile or self.profileSelect.get()
            bot = next(
                (b for b in self.controller.gameEnv.bots if getattr(b, "profileName", "") == profile),
                None,
            )
            success = bool(getattr(bot, "lastEpisodeSuccess", False))
            status = "Success" if success else "Failure"
            self.logOutput.insert(tk.END, f"Completed round {completedRounds}/{totalRounds} {status}\n")
            self.logOutput.see(tk.END)
            self._logLastRound = completedRounds
        if completedRounds == totalRounds:
            self.logOutput.insert(tk.END, "Training completed.\n")
            self.logOutput.see(tk.END)
            self.trainingActive = False
            self.setControlsEnabled(True)
            self._stopStepPoll()
            self._lastWarmupState = None
            self._lastEpsilonValue = None
            self._lastMazeSize = None
            self.stepCounterLabel.configure(text="Episode Steps: 0")
            self.epsilonLabel.configure(text="Epsilon: n/a")

    def cancelTrainingPoll(self) -> None:
        self._stopStepPoll()

    def _startStepPoll(self) -> None:
        self._stopStepPoll()
        self._pollCurrentEpisodeSteps()

    def _stopStepPoll(self) -> None:
        if self._stepPollAfterId is not None:
            try:
                self.after_cancel(self._stepPollAfterId)
            except Exception:
                pass
            self._stepPollAfterId = None

    def _pollCurrentEpisodeSteps(self) -> None:
        if not self.trainingActive:
            self.stepCounterLabel.configure(text="Episode Steps: 0")
            self._stepPollAfterId = None
            return
        steps = 0
        try:
            profile = self.controller.trainingController.activeProfile or self.profileSelect.get()
            if profile:
                bot = next((b for b in self.controller.gameEnv.bots if getattr(b, "profileName", "") == profile), None)
                if bot is not None:
                    self._updateMazeSizeLog()
                    steps = int(getattr(bot, "currentEpisodeSteps", 0))
                    self._updateWarmupLog(bot)
                    self._updateEpsilonDisplay(bot)
        except Exception:
            steps = 0
        self.stepCounterLabel.configure(text=f"Episode Steps: {steps}")
        self._stepPollAfterId = self.after(150, self._pollCurrentEpisodeSteps)

    def _updateWarmupLog(self, bot: Any) -> None:
        warmupActive = False
        try:
            config = getattr(bot, "config", None)
            agent = getattr(bot, "agent", None)
            if config is not None and agent is not None:
                replayWarmupSteps = int(getattr(config, "replayWarmupSteps", 0))
                replaySize = len(getattr(agent, "replay", []))
                warmupActive = bool(getattr(bot, "_collectingWarmup", False)) and replaySize < replayWarmupSteps
        except Exception:
            warmupActive = False

        if self._lastWarmupState is None:
            self._lastWarmupState = warmupActive
            if warmupActive:
                self.logOutput.insert(tk.END, "Warmup entered.\n")
                self.logOutput.see(tk.END)
            return

        if warmupActive != self._lastWarmupState:
            self._lastWarmupState = warmupActive
            self.logOutput.insert(tk.END, "Warmup entered.\n" if warmupActive else "Warmup completed.\n")
            self.logOutput.see(tk.END)

    def _updateMazeSizeLog(self) -> None:
        try:
            currentSize = self.controller.gameEnv.getMazeSize()
        except Exception:
            return
        if self._lastMazeSize is None:
            self._lastMazeSize = currentSize
            return
        if currentSize != self._lastMazeSize:
            previousSize = self._lastMazeSize
            self._lastMazeSize = currentSize
            self.logOutput.insert(
                tk.END,
                f"Curriculum maze size increased: {previousSize[0]}x{previousSize[1]} -> {currentSize[0]}x{currentSize[1]}\n",
            )
            self.logOutput.see(tk.END)

    def _updateEpsilonDisplay(self, bot: Any) -> None:
        agent = getattr(bot, "agent", None)
        if agent is None or not hasattr(agent, "diagnostics"):
            self.epsilonLabel.configure(text="Epsilon: n/a")
            return
        try:
            epsilon = float(agent.diagnostics().epsilon)
            override = getattr(agent, "manualEpsilonOverride", None)
            suffix = " (manual)" if override is not None else ""
            self.epsilonLabel.configure(text=f"Epsilon: {epsilon:.4f}{suffix}")
            if self._lastEpsilonValue is None or abs(epsilon - self._lastEpsilonValue) >= 0.01:
                print(f"[DQN] epsilon={epsilon:.4f}{suffix}")
                self._lastEpsilonValue = epsilon
        except Exception:
            self.epsilonLabel.configure(text="Epsilon: n/a")

    def _getActiveBot(self) -> Any | None:
        profile = self.controller.trainingController.activeProfile or self.profileSelect.get()
        if not profile:
            return None
        try:
            return next((b for b in self.controller.gameEnv.bots if getattr(b, "profileName", "") == profile), None)
        except Exception:
            return None

    def applyManualEpsilon(self) -> None:
        raw = self.epsilonEntry.get().strip()
        if not raw:
            messagebox.showerror("Invalid Epsilon", "Enter a value between 0 and 1.")
            return
        try:
            value = float(raw)
        except Exception:
            messagebox.showerror("Invalid Epsilon", "Manual epsilon must be numeric.")
            return
        bot = self._getActiveBot()
        agent = getattr(bot, "agent", None) if bot is not None else None
        if agent is None or not hasattr(agent, "setManualEpsilon"):
            messagebox.showerror("Unsupported", "Active profile is not a DQN bot.")
            return
        try:
            agent.setManualEpsilon(value)
            self.logOutput.insert(tk.END, f"Manual epsilon override applied: {value:.4f}\n")
            self.logOutput.see(tk.END)
            print(f"[DQN] manual epsilon override applied: {value:.4f}")
        except Exception as exc:
            messagebox.showerror("Invalid Epsilon", str(exc))

    def clearManualEpsilon(self) -> None:
        bot = self._getActiveBot()
        agent = getattr(bot, "agent", None) if bot is not None else None
        if agent is None or not hasattr(agent, "setManualEpsilon"):
            messagebox.showerror("Unsupported", "Active profile is not a DQN bot.")
            return
        try:
            agent.setManualEpsilon(None)
            self.logOutput.insert(tk.END, "Manual epsilon override cleared.\n")
            self.logOutput.see(tk.END)
            print("[DQN] manual epsilon override cleared")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def openVisualization(self) -> None:
        from ui.frames.visualization import VisualizationWindow

        selectedProfile = self.profileSelect.get()
        if not selectedProfile:
            messagebox.showerror("Error", "No profile selected.")
            return
        profileIndex = next(
            (
                i
                for i, bot in enumerate(self.controller.gameEnv.bots)
                if getattr(bot, "profileName", None) == selectedProfile
            ),
            -1,
        )
        if profileIndex == -1:
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
