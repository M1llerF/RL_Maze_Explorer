import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any, Callable, Optional, cast

from bots.dqnlearning.config import DQNConfig
from ui.event_bus import PROFILE_SAVED, PROFILE_DELETED
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
        self._warmupCompatible = False
        self._lastSavedEpisode: int | None = None

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
        self.randomLengthFrame = ttk.Frame(content)
        self.randomMinGenerationLengthVar = tk.StringVar()
        self.randomMaxGenerationLengthVar = tk.StringVar()
        ttk.Label(self.randomLengthFrame, text="Min Generation Length:").pack(side=tk.LEFT)
        self.randomMinGenerationLengthEntry = ttk.Entry(
            self.randomLengthFrame,
            textvariable=self.randomMinGenerationLengthVar,
            width=8,
        )
        self.randomMinGenerationLengthEntry.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(self.randomLengthFrame, text="Max Generation Length:").pack(side=tk.LEFT)
        self.randomMaxGenerationLengthEntry = ttk.Entry(
            self.randomLengthFrame,
            textvariable=self.randomMaxGenerationLengthVar,
            width=8,
        )
        self.randomMaxGenerationLengthEntry.pack(side=tk.LEFT, padx=(6, 0))
        self.poolConfigFrame = ttk.Frame(content)
        ttk.Label(self.poolConfigFrame, text="Pool Size:").pack(side=tk.LEFT)
        self.poolSizeVar = tk.StringVar(value="20")
        self.poolSizeEntry = ttk.Entry(self.poolConfigFrame, textvariable=self.poolSizeVar, width=8)
        self.poolSizeEntry.pack(side=tk.LEFT, padx=(6, 12))
        self.poolSourceLabel = ttk.Label(
            self.poolConfigFrame,
            text="Pool source: generated from random mazes at training start.",
        )
        self.poolSourceLabel.pack(side=tk.LEFT)
        self.mazeSourceHint = ttk.Label(content, text="", foreground="#555555")
        self.mazeSourceHint.pack(pady=(2, 0))

        def onMazeModeChange(event: Any = None) -> None:
            mode = self.mazeMode.get()
            if mode == "Fixed (Builder)":
                fixedMazeConfigured = self.controller.gameEnv.hasFixedMazeConfigured()
                if not fixedMazeConfigured and not self._chooseFixedMaze():
                    self.mazeMode.set("Random")
            self._updateRandomLengthControls()

        self.mazeMode.bind("<<ComboboxSelected>>", onMazeModeChange)
        self.profileSelect.bind("<<ComboboxSelected>>", lambda _e: self._updateWarmupStatus())

        self.mazeButtons = ttk.Frame(content)
        self.mazeButtons.pack(pady=6)
        self.openBuilderBtn = ttk.Button(self.mazeButtons, text="Open Maze Builder", command=lambda: self.controller.showMazeBuilder())
        self.openBuilderBtn.pack(side=tk.LEFT, padx=3)
        self.chooseFixedMazeBtn = ttk.Button(
            self.mazeButtons,
            text="Choose Fixed Maze File",
            command=self._chooseFixedMaze,
        )
        self.chooseFixedMazeBtn.pack(side=tk.LEFT, padx=3)
        self._updateRandomLengthControls()

        # ── Warmup store ────────────────────────────────────────────────
        self.warmupFrame = ttk.LabelFrame(content, text="Warmup Store", padding=(8, 4))
        self.warmupFrame.pack(pady=(6, 0), fill=tk.X, padx=20)

        self.warmupStatusLabel = ttk.Label(self.warmupFrame, text="No warmup store")
        self.warmupStatusLabel.pack()

        collectModeRow = ttk.Frame(self.warmupFrame)
        collectModeRow.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(collectModeRow, text="Collect by:").pack(side=tk.LEFT)
        self.warmupCollectModeVar = tk.StringVar(value="transitions")
        self.warmupTransitionsRadio = ttk.Radiobutton(
            collectModeRow, text="Transitions", variable=self.warmupCollectModeVar,
            value="transitions", command=self._onWarmupCollectModeChanged,
        )
        self.warmupTransitionsRadio.pack(side=tk.LEFT, padx=(6, 0))
        self.warmupCompletionsRadio = ttk.Radiobutton(
            collectModeRow, text="Completions", variable=self.warmupCollectModeVar,
            value="completions", command=self._onWarmupCollectModeChanged,
        )
        self.warmupCompletionsRadio.pack(side=tk.LEFT, padx=(4, 0))

        warmupAmountRow = ttk.Frame(self.warmupFrame)
        warmupAmountRow.pack(fill=tk.X, padx=4, pady=(2, 0))
        self.warmupAmountLabel = ttk.Label(warmupAmountRow, text="Transitions:")
        self.warmupAmountLabel.pack(side=tk.LEFT)
        self.warmupTransitionsVar = tk.StringVar(value=str(DQNConfig.replayWarmupSteps))
        self.warmupTransitionsEntry = ttk.Entry(warmupAmountRow, textvariable=self.warmupTransitionsVar, width=10)
        self.warmupTransitionsEntry.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(self.warmupFrame, text="Warmup Source:").pack(pady=(4, 0))
        self.warmupSourceSelect = ttk.Combobox(self.warmupFrame, state="disabled")
        self.warmupSourceSelect.pack(fill=tk.X, padx=4)
        self.warmupSourceSelect.bind("<<ComboboxSelected>>", lambda _e: self._onWarmupSourceChanged())

        warmupBtns = ttk.Frame(self.warmupFrame)
        warmupBtns.pack(pady=(4, 0))
        self.collectWarmupBtn = ttk.Button(warmupBtns, text="Collect New", command=self._collectWarmup)
        self.collectWarmupBtn.pack(side=tk.LEFT, padx=4)
        self.clearWarmupBtn = ttk.Button(warmupBtns, text="Clear", command=self._clearWarmup)
        self.clearWarmupBtn.pack(side=tk.LEFT, padx=4)

        self.useWarmupVar = tk.BooleanVar(value=False)
        self.useWarmupCheck = ttk.Checkbutton(
            self.warmupFrame,
            text="Inject warmup at training start",
            variable=self.useWarmupVar,
        )
        self.useWarmupCheck.pack(pady=(4, 0))
        self.useWarmupCheck.state(["disabled"])
        self._warmupInteractiveWidgets: list[Any] = [
            self.warmupTransitionsRadio,
            self.warmupCompletionsRadio,
            self.warmupTransitionsEntry,
            self.warmupSourceSelect,
            self.collectWarmupBtn,
            self.clearWarmupBtn,
        ]
        # ────────────────────────────────────────────────────────────────

        actions = ttk.Frame(content)
        actions.pack(pady=10)
        self.startBtn = ttk.Button(actions, text="Start Training", command=self.startTraining)
        self.startBtn.pack(side=tk.LEFT, padx=6)
        self.stopBtn = ttk.Button(actions, text="Stop", command=self.stopTraining, state="disabled")
        self.stopBtn.pack(side=tk.LEFT, padx=6)
        self.resetProfileBtn = ttk.Button(
            actions,
            text="Reset Training Data",
            command=self._resetSelectedProfileTrainingData,
        )
        self.resetProfileBtn.pack(side=tk.LEFT, padx=6)
        self.trainingProgress = ttk.Progressbar(content, orient="horizontal", length=200, mode="determinate")
        self.trainingProgress.pack(pady=10)
        self.stepCounterLabel = ttk.Label(content, text="Episode Steps: 0")
        self.stepCounterLabel.pack()
        self.epsilonLabel = ttk.Label(content, text="Epsilon: n/a")
        self.epsilonLabel.pack()
        self.saveStatusLabel = ttk.Label(content, text="Save Status: n/a")
        self.saveStatusLabel.pack()
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
        logScrollbar = ttk.Scrollbar(logFrame, orient="vertical", command=cast(Callable[..., None], self.logOutput.yview))
        self.logOutput.configure(yscrollcommand=logScrollbar.set)
        self.logOutput.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        logScrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(content, text="Open Visualization", command=self.openVisualization).pack(pady=10)
        self.statusHint = tk.Label(content, text="", fg="#805b00")
        self.statusHint.pack(pady=(0, 6))

        self.loadProfiles()

        def _reloadProfiles(**_: Any) -> None:
            self.loadProfiles()
        controller.eventBus.subscribe(PROFILE_SAVED, _reloadProfiles)
        controller.eventBus.subscribe(PROFILE_DELETED, _reloadProfiles)

    def on_show(self) -> None:
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
                status = bot.getStatus()
                self.warmupStatusLabel.configure(
                    text=f"Collecting… {status.replay_size}/{status.warmup_required} transitions"
                )
        except Exception:
            pass
        self._collectPollAfterId = self.after(150, self._pollCollectProgress)

    # ── Warmup helpers ───────────────────────────────────────────────────────

    def _updateWarmupStatus(self) -> None:
        """Refresh the warmup status label and checkbox state."""
        profile = self.profileSelect.get() or self.lastSelectedProfile
        self.controller.appState.selected_profile = profile

        result = self.controller.gameEnv.warmupService.getWarmupStatus(profile)

        if not result.is_compatible:
            self._warmupCompatible = False
            self.warmupStatusLabel.configure(text=result.status_message)
            self._loadedWarmupStore = None
            self._loadedWarmupSourceProfile = None
            self._warmupOptionsByLabel = {}
            self.warmupSourceSelect.set("")
            self.warmupSourceSelect.configure(state="disabled")
            self.warmupSourceSelect["values"] = ()
            self.useWarmupVar.set(False)
            self._applyWarmupCompatibilityState()
            return

        self._warmupCompatible = True

        if not result.options:
            self.warmupStatusLabel.configure(text=result.status_message)
            self._loadedWarmupStore = None
            self._loadedWarmupSourceProfile = None
            self._warmupOptionsByLabel = {}
            self.warmupSourceSelect.set("")
            self.warmupSourceSelect.configure(state="disabled")
            self.warmupSourceSelect["values"] = ()
            self.useWarmupCheck.state(["disabled"])
            self.useWarmupVar.set(False)
            self._applyWarmupCompatibilityState()
            return

        options = result.options
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
        self._applyWarmupCompatibilityState()

    def _applyWarmupCompatibilityState(self) -> None:
        interactiveState = "normal" if self._warmupCompatible else "disabled"
        for widget in self._warmupInteractiveWidgets:
            try:
                if widget is self.warmupSourceSelect:
                    widget.configure(
                        state=(
                            "readonly"
                            if self._warmupCompatible and bool(self._warmupOptionsByLabel)
                            else "disabled"
                        )
                    )
                else:
                    widget.configure(state=interactiveState)
            except Exception:
                pass
        try:
            if self._warmupCompatible and bool(self._warmupOptionsByLabel):
                self.useWarmupCheck.state(["!disabled"])
            else:
                self.useWarmupCheck.state(["disabled"])
        except Exception:
            pass

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
        randomGenerationRange = self._parseRandomGenerationLengthRange(mode)
        if randomGenerationRange is None:
            return
        minGenerationLength, maxGenerationLength = randomGenerationRange
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
            randomMinGenerationLength=minGenerationLength,
            randomMaxGenerationLength=maxGenerationLength,
            nTransitions=nTransitions,
            nCompletions=nCompletions,
            onProgress=onProgress,
            onComplete=onComplete,
            onError=onError,
        )

    def _clearWarmup(self) -> None:
        sourceProfile = self._loadedWarmupSourceProfile or self.profileSelect.get() or self.lastSelectedProfile
        if not sourceProfile:
            return
        if not self.controller.gameEnv.warmupService.hasStore(sourceProfile):
            self._updateWarmupStatus()
            return
        if messagebox.askyesno("Clear Warmup", f"Delete the warmup store for '{sourceProfile}'?"):
            self.controller.gameEnv.warmupService.deleteStore(sourceProfile)
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
        randomGenerationRange = self._parseRandomGenerationLengthRange(mode)
        if randomGenerationRange is None:
            return
        minGenerationLength, maxGenerationLength = randomGenerationRange
        if mode == "Fixed (Builder)":
            if not self.controller.gameEnv.hasFixedMazeConfigured():
                messagebox.showerror("No Fixed Maze", "No fixed maze set. Open Maze Builder to create or load one, then click 'Use In Training'.")
                return

        self.trainingProgress['maximum'] = rounds
        self.trainingProgress['value'] = 0
        self._logInterval = max(1, rounds // 100)
        self._logLastRound = 0
        self.logOutput.delete("1.0", tk.END)
        self.logOutput.insert(tk.END, f"Training started for {selectedProfile} with {rounds} rounds...\n")
        self._lastSavedEpisode = None
        self.lastSelectedProfile = selectedProfile
        self._lastWarmupState = None
        self._lastEpsilonValue = None
        self.stepCounterLabel.configure(text="Episode Steps: 0")
        self.epsilonLabel.configure(text="Epsilon: n/a")
        self.saveStatusLabel.configure(text="Save Status: waiting for first autosave")
        self.setControlsEnabled(False)
        self.trainingActive = True
        self._lastMazeSize = self.controller.gameEnv.getMazeSize()
        self._startStepPoll()
        self.stopBtn.configure(state="normal")

        def onProgress(done: int, total: int):
            self.controller.root.after(0, self.updateProgress, done, total)

        def onError(err: Exception) -> None:
            def _report():
                self.logOutput.insert(tk.END, f"Training error: {type(err).__name__}: {err}\n")
                self.logOutput.see(tk.END)
                self.trainingActive = False
                self.setControlsEnabled(True)
                self._stopStepPoll()
                self._lastWarmupState = None
                self._lastEpsilonValue = None
                self._lastMazeSize = None
                self.stepCounterLabel.configure(text="Episode Steps: 0")
                self.epsilonLabel.configure(text="Epsilon: n/a")
                self.saveStatusLabel.configure(text="Save Status: n/a")
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
                self.saveStatusLabel.configure(text="Save Status: n/a")
                self.stopBtn.configure(state="disabled")
            self.controller.root.after(0, _done)

        poolSize = self._parsePoolSize() if mode == "Pool" else 0
        if mode == "Pool":
            if poolSize is None:
                self.trainingActive = False
                self.setControlsEnabled(True)
                self._stopStepPoll()
                self.stopBtn.configure(state="disabled")
                self.saveStatusLabel.configure(text="Save Status: n/a")
                return
            self.logOutput.insert(
                tk.END,
                f"Pool mode: training will cycle through {poolSize} auto-generated random mazes.\n",
            )
            self.logOutput.see(tk.END)
        warmupStore = self._loadedWarmupStore if self.useWarmupVar.get() else None
        warmupTransitionCount = self._parseWarmupAmount("transitions") if warmupStore is not None else None
        if warmupStore is not None and warmupTransitionCount is None:
            return
        self.controller.trainingController.start(
            profileName=selectedProfile,
            rounds=rounds,
            mazeMode=mode,
            poolSize=poolSize or 20,
            randomMinGenerationLength=minGenerationLength,
            randomMaxGenerationLength=maxGenerationLength,
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

    def _chooseFixedMaze(self) -> bool:
        path = filedialog.askopenfilename(
            title="Choose a maze JSON to use as Fixed",
            initialdir='mazes',
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return False
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            self.controller.gameEnv.setFixedMaze(state)
            messagebox.showinfo("Fixed Maze Set", f"Using {os.path.basename(path)} for training.")
            return True
        except Exception as e:
            messagebox.showerror("Invalid Maze", f"Could not load maze: {e}")
            return False

    def _resetSelectedProfileTrainingData(self) -> None:
        profileName = self.profileSelect.get()
        if not profileName:
            messagebox.showerror("Error", "No profile selected.")
            return
        if self.trainingActive or self.controller.trainingController.isActive() or self.controller.trainingController.isCollecting():
            messagebox.showinfo("Busy", "Stop the current training or warmup run first.")
            return
        if not messagebox.askyesno(
            "Reset Training Data",
            f"Clear all saved training data for '{profileName}'? This keeps the profile config but removes learned progress.",
        ):
            return
        try:
            self.controller.gameEnv.resetProfileTrainingData(profileName)
            self._loadedWarmupStore = None
            self._loadedWarmupSourceProfile = None
            self._lastWarmupState = None
            self._lastEpsilonValue = None
            self.stepCounterLabel.configure(text="Episode Steps: 0")
            self.epsilonLabel.configure(text="Epsilon: n/a")
            self.useWarmupVar.set(False)
            self.logOutput.insert(tk.END, f"Reset training data for {profileName}.\n")
            self.logOutput.see(tk.END)
            self.loadProfiles()
            self.profileSelect.set(profileName)
            self.lastSelectedProfile = profileName
            self._updateWarmupStatus()
        except Exception as exc:
            messagebox.showerror("Reset Failed", str(exc))

    def _updateRandomLengthControls(self) -> None:
        mode = self.mazeMode.get() or "Random"
        if mode == "Random":
            if not self.randomLengthFrame.winfo_ismapped():
                self.randomLengthFrame.pack(before=self.mazeButtons, pady=(6, 0))
        else:
            if self.randomLengthFrame.winfo_ismapped():
                self.randomLengthFrame.pack_forget()
        if mode == "Pool":
            if not self.poolConfigFrame.winfo_ismapped():
                self.poolConfigFrame.pack(before=self.mazeButtons, pady=(6, 0))
            self.mazeSourceHint.configure(
                text="Pool mode builds a fixed set of random mazes at training start and cycles through them."
            )
        else:
            if self.poolConfigFrame.winfo_ismapped():
                self.poolConfigFrame.pack_forget()
            if mode == "Fixed (Builder)":
                self.mazeSourceHint.configure(
                    text="Fixed mode reuses the maze currently selected from the builder or file chooser."
                )
            else:
                self.mazeSourceHint.configure(
                    text="Random mode generates a new maze each reset. Min/Max Generation Length constrain path length."
                )

    def _parsePoolSize(self) -> int | None:
        raw = self.poolSizeVar.get().strip()
        if not raw:
            messagebox.showerror("Invalid Pool Size", "Pool size must be a positive integer.")
            return None
        try:
            value = int(raw)
        except Exception:
            messagebox.showerror("Invalid Pool Size", "Pool size must be a positive integer.")
            return None
        if value <= 0:
            messagebox.showerror("Invalid Pool Size", "Pool size must be greater than zero.")
            return None
        return value

    def _parseRandomGenerationLengthRange(self, mode: str) -> tuple[int | None, int | None] | None:
        if mode != "Random":
            return (None, None)
        rawMin = self.randomMinGenerationLengthVar.get().strip()
        rawMax = self.randomMaxGenerationLengthVar.get().strip()
        minValue: int | None = None
        maxValue: int | None = None
        if rawMin:
            try:
                minValue = int(rawMin)
            except Exception:
                messagebox.showerror("Invalid Generation Length", "Minimum generation length must be an integer.")
                return None
            if minValue < 0:
                messagebox.showerror("Invalid Generation Length", "Minimum generation length must be 0 or greater.")
                return None
        if rawMax:
            try:
                maxValue = int(rawMax)
            except Exception:
                messagebox.showerror("Invalid Generation Length", "Maximum generation length must be an integer.")
                return None
            if maxValue < 0:
                messagebox.showerror("Invalid Generation Length", "Maximum generation length must be 0 or greater.")
                return None
        if minValue is not None and maxValue is not None and minValue > maxValue:
            messagebox.showerror(
                "Invalid Generation Length",
                "Minimum generation length must be less than or equal to maximum generation length.",
            )
            return None
        return (minValue, maxValue)

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
            self.randomMinGenerationLengthEntry.configure(state=entryState)
            self.randomMaxGenerationLengthEntry.configure(state=entryState)
        except Exception:
            pass
        try:
            self.openBuilderBtn.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            pass
        try:
            self.chooseFixedMazeBtn.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            pass
        try:
            self.resetProfileBtn.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            pass
        if enabled:
            self._applyWarmupCompatibilityState()
        else:
            for widget in self._warmupInteractiveWidgets:
                try:
                    widget.configure(state="disabled")
                except Exception:
                    pass
            try:
                self.useWarmupCheck.state(["disabled"])
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
        self.saveStatusLabel.configure(text="Save Status: n/a")

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
            _OUTCOME_LABELS = {
                "goal_reached": "Goal reached",
                "death_by_enemy": "Caught by enemy",
                "step_limit": "Timed out",
                "step_limit_loop": "Timed out",
                "step_limit_statistics": "Timed out",
                "no_progress": "No progress",
                "stop_requested": "Stopped",
            }
            if bot is not None:
                outcome = str(getattr(bot, "lastEpisodeOutcome", "") or "")
                wallHits = int(getattr(bot, "lastEpisodeWallHits", 0))
                kills = int(getattr(bot, "lastEpisodeEnemyKills", 0))
                steps = int(getattr(bot, "lastEpisodeSteps", 0))
                outcomeText = _OUTCOME_LABELS.get(
                    outcome,
                    outcome.replace("_", " ").title() if outcome else (
                        "Goal reached" if bot.getStatus().last_episode_success else "Failed"
                    ),
                )
            else:
                outcomeText, wallHits, kills, steps = "?", 0, 0, 0
            parts = [f"Round {completedRounds}/{totalRounds}", outcomeText, f"steps: {steps}", f"wall hits: {wallHits}"]
            if kills:
                parts.append(f"kills: {kills}")
            self.logOutput.insert(tk.END, "  |  ".join(parts) + "\n")
            self.logOutput.see(tk.END)
            self._logLastRound = completedRounds
            if bot is not None:
                self._updateSaveStatus(bot, completedRounds)
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
            self.saveStatusLabel.configure(text="Save Status: n/a")

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
                    steps = bot.getStatus().current_episode_steps
                    self._updateWarmupLog(bot)
                    self._updateEpsilonDisplay(bot)
        except Exception:
            steps = 0
        self.stepCounterLabel.configure(text=f"Episode Steps: {steps}")
        self._stepPollAfterId = self.after(150, self._pollCurrentEpisodeSteps)

    def _updateWarmupLog(self, bot: Any) -> None:
        try:
            warmupActive = bot.getStatus().is_warming_up
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
        try:
            status = bot.getStatus()
            suffix = " (manual)" if status.epsilon_is_manual else ""
            self.epsilonLabel.configure(text=f"Epsilon: {status.current_exploration_rate:.4f}{suffix}")
            self._lastEpsilonValue = status.current_exploration_rate
        except Exception:
            self.epsilonLabel.configure(text="Epsilon: n/a")

    def _updateSaveStatus(self, bot: Any, completedRounds: int) -> None:
        try:
            status = bot.getStatus()
        except Exception:
            return
        profileName = str(getattr(bot, "profileName", ""))
        label = status.artifact_save_label

        if label == "Q-table":
            if self._lastSavedEpisode != completedRounds:
                self._lastSavedEpisode = completedRounds
                self.logOutput.insert(tk.END, f"Autosaved {profileName} after round {completedRounds} ({label}).\n")
                self.logOutput.see(tk.END)
            self.saveStatusLabel.configure(text=f"Save Status: autosaved after round {completedRounds} ({label})")
            return

        freq = status.checkpoint_frequency
        count = status.episode_count
        if freq > 0 and count > 0 and count % freq == 0:
            if self._lastSavedEpisode != count:
                self._lastSavedEpisode = count
                self.logOutput.insert(tk.END, f"Autosaved {profileName} after round {count} ({label}).\n")
                self.logOutput.see(tk.END)
            self.saveStatusLabel.configure(text=f"Save Status: autosaved after round {count} ({label})")
        elif freq > 0:
            nextSave = freq * ((count // freq) + 1)
            self.saveStatusLabel.configure(text=f"Save Status: next {label} at round {nextSave}")
        else:
            self.saveStatusLabel.configure(text="Save Status: checkpoint autosave disabled")

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
        if bot is None or not bot.setManualEpsilon(value):
            messagebox.showerror("Unsupported", "Active profile is not a DQN bot.")
            return
        self.logOutput.insert(tk.END, f"Manual epsilon override applied: {value:.4f}\n")
        self.logOutput.see(tk.END)

    def clearManualEpsilon(self) -> None:
        bot = self._getActiveBot()
        if bot is None or not bot.setManualEpsilon(None):
            messagebox.showerror("Unsupported", "Active profile is not a DQN bot.")
            return
        self.logOutput.insert(tk.END, "Manual epsilon override cleared.\n")
        self.logOutput.see(tk.END)

    def openVisualization(self) -> None:
        from ui.frames.visualization import VisualizationWindow

        selectedProfile = self.profileSelect.get()
        if not selectedProfile:
            messagebox.showerror("Error", "No profile selected.")
            return
        mode = self.mazeMode.get() or "Random"
        randomGenerationRange = self._parseRandomGenerationLengthRange(mode)
        if randomGenerationRange is None:
            return
        minGenerationLength, maxGenerationLength = randomGenerationRange
        try:
            if mode == "Random":
                self.controller.gameEnv.setRandomGenerationLengthRange(
                    minGenerationLength,
                    maxGenerationLength,
                )
            else:
                self.controller.gameEnv.setRandomGenerationLengthRange(None, None)
        except Exception as exc:
            messagebox.showerror("Invalid Generation Length", str(exc))
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
