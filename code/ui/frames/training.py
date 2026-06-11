import json
import os
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Any, Callable, Optional, cast

from bots.dqnlearning.config import DQNConfig
from ui.eventBus import FIXED_MAZE_SELECTED, PROFILE_SAVED, PROFILE_DELETED
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
        self._lastEvaluationRound: int | None = None
        self._pendingWarmupClearProfile: str | None = None
        self._pendingTrainingResetProfile: str | None = None

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

        self.mazeModeLabel = ttk.Label(content, text="Maze Source: Random")
        self.mazeModeLabel.pack(pady=(10, 0))
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
        self.randomEnemyFrame = ttk.Frame(content)
        self.randomIncludeEnemiesVar = tk.BooleanVar(value=False)
        self.randomIncludeEnemiesCheck = ttk.Checkbutton(
            self.randomEnemyFrame,
            text="Generate enemies",
            variable=self.randomIncludeEnemiesVar,
        )
        self.randomIncludeEnemiesCheck.pack(side=tk.LEFT)
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
        self.profileSelect.bind("<<ComboboxSelected>>", self._onProfileSelected)

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
        self.statusVar = tk.StringVar(value="Ready.")
        self.statusLabel = tk.Label(
            content,
            textvariable=self.statusVar,
            fg="#374151",
            justify=tk.LEFT,
            anchor="w",
            wraplength=920,
        )
        self.statusLabel.pack(fill=tk.X, padx=20, pady=(4, 0))
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
        controller.eventBus.subscribe(FIXED_MAZE_SELECTED, self._handleFixedMazeSelected)

    def on_show(self) -> None:
        self.loadProfiles()
        self._updateVisualizationHint()

    def loadProfiles(self) -> None:
        profiles = self.controller.gameEnv.profileManager.listProfiles()
        self.profileSelect['values'] = profiles
        if self.lastSelectedProfile and self.lastSelectedProfile in profiles:
            try:
                self.profileSelect.set(self.lastSelectedProfile)
            except Exception:
                pass
        elif self.profileSelect.get() not in profiles:
            try:
                self.profileSelect.set("")
            except Exception:
                pass
        self._resetPendingActions()
        self._updateWarmupStatus()
        self._updateRandomLengthControls()

    def _onProfileSelected(self, event: Any = None) -> None:
        del event
        self._resetPendingActions()
        self._updateWarmupStatus()

    def _setStatus(self, message: str, *, tone: str = "info") -> None:
        colors = {
            "info": "#374151",
            "success": "#166534",
            "warning": "#805b00",
            "error": "#b91c1c",
        }
        self.statusVar.set(message)
        self.statusLabel.configure(fg=colors.get(tone, colors["info"]))

    def _appendLog(self, message: str) -> None:
        self.logOutput.insert(tk.END, message.rstrip("\n") + "\n")
        self.logOutput.see(tk.END)

    def _resetWarmupClearConfirmation(self) -> None:
        self._pendingWarmupClearProfile = None
        try:
            self.clearWarmupBtn.configure(text="Clear")
        except Exception:
            pass

    def _resetTrainingResetConfirmation(self) -> None:
        self._pendingTrainingResetProfile = None
        try:
            self.resetProfileBtn.configure(text="Reset Training Data")
        except Exception:
            pass

    def _resetPendingActions(self) -> None:
        self._resetWarmupClearConfirmation()
        self._resetTrainingResetConfirmation()

    def _handleFixedMazeSelected(self, *, label: str = "") -> None:
        self.mazeMode.set("Fixed (Builder)")
        self._updateRandomLengthControls()

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
            self._setStatus("Select a profile before collecting warmup.", tone="error")
            return
        if self.trainingActive or self.controller.trainingController.isCollecting():
            self._setStatus("Wait for the current training or warmup run to finish.", tone="warning")
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
        self._appendLog(f"Collecting warmup for {profile}...")
        self._setStatus(f"Collecting warmup for {profile}.", tone="info")
        self.warmupStatusLabel.configure(text=f"Collecting… 0 {unitLabel}")

        def onProgress(current: int, total: int) -> None:
            def _update() -> None:
                self.warmupStatusLabel.configure(text=f"Collecting… {current}/{total} {unitLabel}")
            self.controller.root.after(0, _update)

        def onComplete(store: Any) -> None:
            def _finish() -> None:
                self._stopCollectPoll()
                self._appendLog(f"Warmup collected: {store.transitionCount} transitions - saved.")
                self._setStatus("Warmup collection finished and saved.", tone="success")
                self.collectWarmupBtn.configure(state="normal")
                self.clearWarmupBtn.configure(state="normal")
                self.startBtn.configure(state="normal")
                self._updateWarmupStatus()
            self.controller.root.after(0, _finish)

        def onError(exc: Exception) -> None:
            def _report() -> None:
                self._stopCollectPoll()
                self._appendLog(f"Warmup collection failed: {exc}")
                self._setStatus(f"Warmup collection failed: {exc}", tone="error")
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
            randomIncludeEnemies=self.randomIncludeEnemiesVar.get(),
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
            self._resetWarmupClearConfirmation()
            self._updateWarmupStatus()
            self._setStatus(f"No warmup store exists for '{sourceProfile}'.", tone="info")
            return
        if self._pendingWarmupClearProfile != sourceProfile:
            self._resetTrainingResetConfirmation()
            self._pendingWarmupClearProfile = sourceProfile
            self.clearWarmupBtn.configure(text="Confirm Clear")
            self._setStatus(
                f"Click Clear again to delete the warmup store for '{sourceProfile}'.",
                tone="warning",
            )
            return
        self.controller.gameEnv.warmupService.deleteStore(sourceProfile)
        self._resetWarmupClearConfirmation()
        self._loadedWarmupStore = None
        self._loadedWarmupSourceProfile = None
        self.useWarmupVar.set(False)
        self._updateWarmupStatus()
        self._appendLog(f"Warmup store cleared for {sourceProfile}.")
        self._setStatus(f"Warmup store cleared for '{sourceProfile}'.", tone="success")

    def startTraining(self) -> None:
        if self.trainingActive:
            return
        self._resetPendingActions()
        selectedProfile = self.profileSelect.get()
        if not selectedProfile:
            self._setStatus("Select a profile before starting training.", tone="error")
            return

        roundsTxt = self.roundsEntry.get()
        if not roundsTxt.isdigit():
            self._setStatus("Number of rounds must be a positive integer.", tone="error")
            return
        rounds = int(roundsTxt)

        mode = self.mazeMode.get() or "Random"
        randomGenerationRange = self._parseRandomGenerationLengthRange(mode)
        if randomGenerationRange is None:
            return
        minGenerationLength, maxGenerationLength = randomGenerationRange
        if mode == "Fixed (Builder)":
            if not self.controller.gameEnv.hasFixedMazeConfigured():
                self._setStatus(
                    "No fixed maze is set. Open Maze Builder to create or load one, then click 'Use In Training'.",
                    tone="error",
                )
                return

        self.trainingProgress['maximum'] = rounds
        self.trainingProgress['value'] = 0
        self._logInterval = max(1, rounds // 100)
        self._logLastRound = 0
        self.logOutput.delete("1.0", tk.END)
        self._appendLog(f"Training started for {selectedProfile} with {rounds} rounds...")
        self._lastSavedEpisode = None
        self._lastEvaluationRound = None
        self.lastSelectedProfile = selectedProfile
        self._lastWarmupState = None
        self._lastEpsilonValue = None
        self.stepCounterLabel.configure(text="Episode Steps: 0")
        self.epsilonLabel.configure(text="Epsilon: n/a")
        self.saveStatusLabel.configure(text="Save Status: waiting for first autosave")
        self._setStatus(f"Training started for {selectedProfile}.", tone="info")
        self.setControlsEnabled(False)
        self.trainingActive = True
        self._lastMazeSize = self.controller.gameEnv.getMazeSize()
        self._startStepPoll()
        self.stopBtn.configure(state="normal")
        evalDefinition = self._getEvaluationDefinition()
        evalFreq = int(evalDefinition.get("frequency", 0))
        if bool(evalDefinition.get("is_dedicated_episode", False)) and evalFreq > 0:
            evalEpsilon = evalDefinition.get("eval_epsilon")
            epsilonText = "n/a" if evalEpsilon is None else f"{float(evalEpsilon):.4f}"
            savedParts: list[str] = []
            if bool(evalDefinition.get("append_reward", False)):
                savedParts.append("reward")
            if bool(evalDefinition.get("save_maze_episode", False)):
                savedParts.append("maze")
            savedText = " + ".join(savedParts) if savedParts else "no snapshots"
            self._appendLog(
                f"Scheduled evaluation: every {evalFreq} rounds at epsilon {epsilonText}; "
                f"evaluation saves {savedText} artifacts and does not create checkpoints."
            )

        def onProgress(done: int, total: int):
            self.controller.root.after(0, self.updateProgress, done, total)

        def onError(err: Exception) -> None:
            def _report():
                self._appendLog(f"Training error: {type(err).__name__}: {err}")
                self._setStatus(f"Training failed: {err}", tone="error")
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
                self._setStatus("Training run finished.", tone="success")
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
            self._appendLog(
                f"Pool mode: training will cycle through {poolSize} auto-generated random mazes."
            )
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
            randomIncludeEnemies=self.randomIncludeEnemiesVar.get(),
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
            self._setStatus(f"Warmup {label} must be a positive integer.", tone="error")
            return None
        try:
            value = int(raw)
        except Exception:
            self._setStatus(f"Warmup {label} must be a positive integer.", tone="error")
            return None
        if value <= 0:
            self._setStatus(f"Warmup {label} must be greater than zero.", tone="error")
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
            label = os.path.basename(path) or "Fixed Maze"
            self.controller.gameEnv.setFixedMaze(state, label=label)
            self.controller.eventBus.emit(FIXED_MAZE_SELECTED, label=label)
            self._appendLog(f"Fixed maze selected: {label}")
            self._setStatus(f"Fixed maze selected: {label}", tone="success")
            return True
        except Exception as e:
            self._setStatus(f"Could not load maze: {e}", tone="error")
            return False

    def _resetSelectedProfileTrainingData(self) -> None:
        profileName = self.profileSelect.get()
        if not profileName:
            self._setStatus("Select a profile before resetting training data.", tone="error")
            return
        if self.trainingActive or self.controller.trainingController.isActive() or self.controller.trainingController.isCollecting():
            self._setStatus("Stop the current training or warmup run first.", tone="warning")
            return
        if self._pendingTrainingResetProfile != profileName:
            self._resetWarmupClearConfirmation()
            self._pendingTrainingResetProfile = profileName
            self.resetProfileBtn.configure(text="Confirm Reset")
            self._setStatus(
                f"Click Reset Training Data again to clear learned progress for '{profileName}'.",
                tone="warning",
            )
            return
        try:
            self.controller.gameEnv.resetProfileTrainingData(profileName)
            self._resetTrainingResetConfirmation()
            self._loadedWarmupStore = None
            self._loadedWarmupSourceProfile = None
            self._lastWarmupState = None
            self._lastEpsilonValue = None
            self.stepCounterLabel.configure(text="Episode Steps: 0")
            self.epsilonLabel.configure(text="Epsilon: n/a")
            self.useWarmupVar.set(False)
            self._appendLog(f"Reset training data for {profileName}.")
            self._setStatus(f"Reset training data for '{profileName}'.", tone="success")
            self.loadProfiles()
            self.profileSelect.set(profileName)
            self.lastSelectedProfile = profileName
            self._updateWarmupStatus()
        except Exception as exc:
            self._resetTrainingResetConfirmation()
            self._setStatus(f"Reset failed: {exc}", tone="error")

    def _updateRandomLengthControls(self) -> None:
        mode = self.mazeMode.get() or "Random"
        if mode == "Random":
            if not self.randomLengthFrame.winfo_ismapped():
                self.randomLengthFrame.pack(before=self.mazeButtons, pady=(6, 0))
        else:
            if self.randomLengthFrame.winfo_ismapped():
                self.randomLengthFrame.pack_forget()
        if mode in {"Random", "Pool"}:
            if not self.randomEnemyFrame.winfo_ismapped():
                self.randomEnemyFrame.pack(before=self.mazeButtons, pady=(4, 0))
        else:
            if self.randomEnemyFrame.winfo_ismapped():
                self.randomEnemyFrame.pack_forget()
        if mode == "Pool":
            if not self.poolConfigFrame.winfo_ismapped():
                self.poolConfigFrame.pack(before=self.mazeButtons, pady=(6, 0))
            self.mazeModeLabel.configure(text="Maze Source: Pool")
            self.mazeSourceHint.configure(
                text="Pool mode builds a fixed set of random mazes at training start and cycles through them."
            )
        else:
            if self.poolConfigFrame.winfo_ismapped():
                self.poolConfigFrame.pack_forget()
            if mode == "Fixed (Builder)":
                fixedLabel = self.controller.gameEnv.getFixedMazeLabel() or "Builder Maze"
                self.mazeModeLabel.configure(text=f"Maze Source: {fixedLabel}")
                self.mazeSourceHint.configure(
                    text=f"Fixed mode reuses the selected maze: {fixedLabel}."
                )
            else:
                self.mazeModeLabel.configure(text="Maze Source: Random")
                self.mazeSourceHint.configure(
                    text="Random mode generates a new maze each reset. Min/Max Generation Length constrain path length."
                )

    def _parsePoolSize(self) -> int | None:
        raw = self.poolSizeVar.get().strip()
        if not raw:
            self._setStatus("Pool size must be a positive integer.", tone="error")
            return None
        try:
            value = int(raw)
        except Exception:
            self._setStatus("Pool size must be a positive integer.", tone="error")
            return None
        if value <= 0:
            self._setStatus("Pool size must be greater than zero.", tone="error")
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
                self._setStatus("Minimum generation length must be an integer.", tone="error")
                return None
            if minValue < 0:
                self._setStatus("Minimum generation length must be 0 or greater.", tone="error")
                return None
        if rawMax:
            try:
                maxValue = int(rawMax)
            except Exception:
                self._setStatus("Maximum generation length must be an integer.", tone="error")
                return None
            if maxValue < 0:
                self._setStatus("Maximum generation length must be 0 or greater.", tone="error")
                return None
        if minValue is not None and maxValue is not None and minValue > maxValue:
            self._setStatus(
                "Minimum generation length must be less than or equal to maximum generation length.",
                tone="error",
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
            self.randomIncludeEnemiesCheck.configure(state=entryState)
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
        self._setStatus("Stop requested. Waiting for the current episode to finish.", tone="warning")

    def updateProgress(self, completedRounds: int, totalRounds: int) -> None:
        self.trainingProgress['value'] = completedRounds
        if (
            completedRounds == totalRounds
            or completedRounds == 1
            or completedRounds % self._logInterval == 0
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
            self._appendLog("  |  ".join(parts))
            self._logLastRound = completedRounds
            if bot is not None:
                self._updateSaveStatus(bot, completedRounds)
                self._logEvaluationStatus(bot, completedRounds)
        if completedRounds == totalRounds:
            self._appendLog("Training completed.")
            self.trainingActive = False
            self.setControlsEnabled(True)
            self._stopStepPoll()
            self._lastWarmupState = None
            self._lastEpsilonValue = None
            self._lastMazeSize = None
            self.stepCounterLabel.configure(text="Episode Steps: 0")
            self.epsilonLabel.configure(text="Epsilon: n/a")
            self.saveStatusLabel.configure(text="Save Status: n/a")
            self._setStatus("Training completed.", tone="success")

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
                self._appendLog("Warmup entered.")
            return

        if warmupActive != self._lastWarmupState:
            self._lastWarmupState = warmupActive
            self._appendLog("Warmup entered." if warmupActive else "Warmup completed.")

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
            self._appendLog(
                f"Curriculum maze size increased: {previousSize[0]}x{previousSize[1]} -> {currentSize[0]}x{currentSize[1]}"
            )

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
        freq = max(0, int(status.checkpoint_frequency))
        count = status.episode_count
        if freq > 0 and count > 0 and count % freq == 0:
            if self._lastSavedEpisode != count:
                self._lastSavedEpisode = count
                self._appendLog(f"Autosaved {profileName} after round {completedRounds} ({label}).")
            self.saveStatusLabel.configure(text=f"Save Status: autosaved after round {completedRounds} ({label})")
        elif freq > 0:
            rounds_to_next = freq if count <= 0 else (freq - (count % freq))
            nextSave = completedRounds + rounds_to_next
            self.saveStatusLabel.configure(text=f"Save Status: next {label} at round {nextSave}")
        else:
            self.saveStatusLabel.configure(text="Save Status: checkpoint autosave disabled")

    def _logEvaluationStatus(self, bot: Any, completedRounds: int) -> None:
        definition = self._getEvaluationDefinition(bot)
        evalFreq = int(definition.get("frequency", 0))
        if (
            not bool(definition.get("is_dedicated_episode", False))
            or evalFreq <= 0
            or completedRounds <= 0
            or completedRounds % evalFreq != 0
        ):
            return
        if self._lastEvaluationRound == completedRounds:
            return
        self._lastEvaluationRound = completedRounds
        success = getattr(bot, "lastEvaluationSuccess", None)
        steps = getattr(bot, "lastEvaluationSteps", None)
        reward = getattr(bot, "lastEvaluationReward", None)
        evalEpsilon = definition.get("eval_epsilon")
        epsilonText = "n/a" if evalEpsilon is None else f"{float(evalEpsilon):.4f}"
        parts = [f"Evaluation after round {completedRounds}", f"epsilon: {epsilonText}"]
        if success is not None:
            parts.append("goal reached" if bool(success) else "failed")
        if steps is not None:
            parts.append(f"steps: {int(steps)}")
        if reward is not None:
            parts.append(f"reward: {float(reward):.2f}")
        savedParts: list[str] = []
        if bool(definition.get("append_reward", False)):
            savedParts.append("reward")
        if bool(definition.get("save_maze_episode", False)):
            savedParts.append("maze")
        parts.append(f"saved: {' + '.join(savedParts) if savedParts else 'none'}")
        parts.append("checkpoint unchanged")
        self._appendLog("  |  ".join(parts))

    def _getActiveBot(self) -> Any | None:
        profile = self.controller.trainingController.activeProfile or self.profileSelect.get()
        if not profile:
            return None
        try:
            return next((b for b in self.controller.gameEnv.bots if getattr(b, "profileName", "") == profile), None)
        except Exception:
            return None

    def _getEvaluationDefinition(self, bot: Any | None = None) -> dict[str, Any]:
        target = bot if bot is not None else self._getActiveBot()
        getter = getattr(target, "getEvaluationEpisodeDefinition", None)
        if callable(getter):
            try:
                definition = getter()
                persistence = getattr(definition, "persistence", None)
                return {
                    "frequency": max(0, int(getattr(definition, "frequency", 0))),
                    "is_dedicated_episode": getattr(definition, "is_dedicated_episode", False),
                    "eval_epsilon": getattr(definition, "eval_epsilon", None),
                    "save_maze_episode": getattr(persistence, "save_maze_episode", False),
                    "append_reward": getattr(persistence, "append_reward", False),
                }
            except Exception:
                pass
        selectedProfile = self.profileSelect.get() or self.lastSelectedProfile
        if selectedProfile:
            try:
                profile = self.controller.gameEnv.profileManager.loadProfile(selectedProfile)
                if str(getattr(profile, "botType", "")) == "DQNBot":
                    freq = int(getattr(getattr(profile, "config", None), "evaluationFrequency", 0))
                    return {
                        "frequency": max(0, freq),
                        "is_dedicated_episode": freq > 0,
                        "eval_epsilon": 0.0,
                        "save_maze_episode": True,
                        "append_reward": True,
                    }
            except Exception:
                pass
        return {
            "frequency": 0,
            "is_dedicated_episode": False,
            "eval_epsilon": None,
            "save_maze_episode": False,
            "append_reward": False,
        }

    def applyManualEpsilon(self) -> None:
        raw = self.epsilonEntry.get().strip()
        if not raw:
            self._setStatus("Enter a manual epsilon value between 0 and 1.", tone="error")
            return
        try:
            value = float(raw)
        except Exception:
            self._setStatus("Manual epsilon must be numeric.", tone="error")
            return
        bot = self._getActiveBot()
        if bot is None or not bot.setManualEpsilon(value):
            self._setStatus("Manual epsilon override is only available for DQN bots.", tone="error")
            return
        self._appendLog(f"Manual epsilon override applied: {value:.4f}")
        self._setStatus(f"Manual epsilon override applied: {value:.4f}", tone="success")

    def clearManualEpsilon(self) -> None:
        bot = self._getActiveBot()
        if bot is None or not bot.setManualEpsilon(None):
            self._setStatus("Manual epsilon override is only available for DQN bots.", tone="error")
            return
        self._appendLog("Manual epsilon override cleared.")
        self._setStatus("Manual epsilon override cleared.", tone="success")

    def openVisualization(self) -> None:
        from ui.frames.visualization import VisualizationWindow

        selectedProfile = self.profileSelect.get()
        if not selectedProfile:
            self._setStatus("Select a profile before opening visualization.", tone="error")
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
            self.controller.gameEnv.setRandomIncludeEnemies(
                mode in {"Random", "Pool"} and self.randomIncludeEnemiesVar.get()
            )
        except Exception as exc:
            self._setStatus(str(exc), tone="error")
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
            self._updateVisualizationHint()
        else:
            self.visualizationWindow = VisualizationWindow(self.controller.root, self.controller.gameEnv, selectedProfile, profileIndex)
            self.visualizationWindow.bind("<Destroy>", lambda _event: self._onVisualizationClosed(), add="+")
            try:
                self._appendLog("Opened visualization. Training is PAUSED while the window is open.")
                self._setStatus("Visualization opened for the selected profile.", tone="success")
                self._updateVisualizationHint()
            except Exception:
                pass

    def _updateVisualizationHint(self) -> None:
        window = self.visualizationWindow
        if window is not None and window.winfo_exists():
            self.statusHint.configure(text="Note: Training is paused while visualization is open.")
        elif str(self.statusHint.cget("text")).startswith("Note: Training is paused"):
            self.statusHint.configure(text="")

    def _onVisualizationClosed(self) -> None:
        self.visualizationWindow = None
        self._updateVisualizationHint()
