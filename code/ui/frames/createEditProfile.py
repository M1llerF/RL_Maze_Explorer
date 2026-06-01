# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, cast

from botConfigs import botConfigs, buildConfigForBotType
from rewardSystem import RewardConfig
from botProfile import BotProfile


class CreateEditProfileFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller
        self.currentConfigWidgets: list[Any] = []
        self.paramVars: dict[str, tk.StringVar] = {}
        self.rewardVars: dict[str, tk.StringVar] = {}
        self.autoVars: dict[str, Any] = {}
        self.paramEntries: dict[str, Any] = {}
        self.rewardEntries: dict[str, Any] = {}
        self.profile: BotProfile | None = None
        try:
            self._style = ttk.Style()
            self._style.configure("Error.TEntry", fieldbackground="#ffecec")
        except Exception:
            self._style = None

        ttk.Label(self, text="Create/Edit Profile", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)

        buttonRow = ttk.Frame(self)
        buttonRow.pack(pady=6)
        ttk.Button(buttonRow, text="Save", command=self.saveProfile).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttonRow, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=6)

        ttk.Label(self, text="Profile Name:").pack()
        self.profileNameEntry = ttk.Entry(self)
        self.profileNameEntry.pack()

        ttk.Label(self, text="Bot Type:").pack()
        self.botTypeEntry = ttk.Combobox(self, values=list(botConfigs.keys()))
        self.botTypeEntry.pack()
        self.botTypeEntry.bind("<<ComboboxSelected>>", self.updateBotConfigUi)

        self.configFrame = ttk.Frame(self)
        self.configFrame.pack(fill="both", expand=True, pady=10)

    def _targetTab(self, paramKey: str, tabs: dict[str, Any]) -> Any:
        key = str(paramKey)
        if key in {"useRichEncoding", "usePositionInState", "neuralMapWidth", "neuralMapHeight", "neuralMapPoolSize"}:
            return tabs["encoding"]
        if key in {
            "warmupEnabled",
            "replayWarmupSteps",
            "immediateReversalPenalty",
            "repeatVisitPenaltyScale",
            "noProgressPenalty",
            "noProgressPatienceFactor",
            "minNoProgressSteps",
            "maxNoProgressSteps",
        }:
            return tabs["warmup"]
        if key in {
            "learningRate",
            "discountFactor",
            "epsilonStart",
            "epsilonEnd",
            "epsilonDecaySteps",
            "replayCapacity",
            "batchSize",
            "trainFrequency",
            "targetUpdateFrequency",
            "hiddenSize",
            "maxStepsPerEpisode",
            "rewardClipMin",
            "rewardClipMax",
            "diagnosticsFrequency",
        }:
            return tabs["core"]
        return tabs["general"]

    # ----- UI building -----
    def updateBotConfigUi(self, event: Any = None) -> None:
        for widget in self.currentConfigWidgets:
            widget.destroy()
        self.currentConfigWidgets.clear()
        self.paramVars.clear()
        self.rewardVars.clear()
        self.autoVars.clear()
        self.paramEntries.clear()
        self.rewardEntries.clear()

        botType = self.botTypeEntry.get()
        if botType not in botConfigs:
            return
        config = botConfigs[botType]

        notebook = ttk.Notebook(self.configFrame)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        tabs = {
            "general": ttk.Frame(notebook),
            "core": ttk.Frame(notebook),
            "encoding": ttk.Frame(notebook),
            "warmup": ttk.Frame(notebook),
            "rewards": ttk.Frame(notebook),
        }
        notebook.add(tabs["general"], text="General")
        notebook.add(tabs["core"], text="DQN Core")
        notebook.add(tabs["encoding"], text="Encoding")
        notebook.add(tabs["warmup"], text="Warmup/Planner")
        notebook.add(tabs["rewards"], text="Rewards")
        self.currentConfigWidgets.extend([notebook, *tabs.values()])

        if "params" in config:
            for paramName, paramKey in config["params"].items():
                host = self._targetTab(paramKey, tabs)
                row = ttk.Frame(host)
                row.pack(fill="x", pady=2, padx=8)
                label = ttk.Label(row, text=f"{paramName}:")
                label.pack(side=tk.LEFT)
                var = tk.StringVar()
                entry = ttk.Entry(row, textvariable=var, width=14)
                entry.pack(side=tk.LEFT, padx=6)
                self.currentConfigWidgets.extend([row, label, entry])
                self.paramVars[paramKey] = var
                self.paramEntries[paramKey] = entry

        if "rewards" in config:
            header = ttk.Label(tabs["rewards"], text="Reward Configuration:")
            header.pack(anchor="w", padx=8, pady=(2, 6))
            self.currentConfigWidgets.append(header)
            for rewardKey, defaultValue in config["rewards"].items():
                row = ttk.Frame(tabs["rewards"])
                row.pack(fill="x", padx=8, pady=2)
                rewardLabel = ttk.Label(row, text=f"{rewardKey}:")
                rewardLabel.pack(side=tk.LEFT)
                var = tk.StringVar(value=defaultValue)
                rewardEntry = ttk.Entry(row, textvariable=var, width=14)
                rewardEntry.pack(side=tk.LEFT, padx=6)
                self.currentConfigWidgets.extend([row, rewardLabel, rewardEntry])
                self.rewardVars[rewardKey] = var
                self.rewardEntries[rewardKey] = rewardEntry

    # ----- Data binding -----
    def loadProfile(self, profile: Any = None) -> None:
        self.profile = profile
        if profile:
            self.profileNameEntry.delete(0, tk.END)
            self.profileNameEntry.insert(0, profile.name)
            self.botTypeEntry.set(profile.botType)
            self.updateBotConfigUi()

            if profile.config:
                for paramKey, var in self.paramVars.items():
                    var.set(getattr(profile.config, paramKey, ""))

            if profile.rewardConfig:
                for rewardKey, var in self.rewardVars.items():
                    var.set(profile.rewardConfig.rewardModifiers.get(rewardKey, ""))

    # ----- Actions -----
    def saveProfile(self) -> None:
        profileName = self.profileNameEntry.get()
        botType = self.botTypeEntry.get()

        if botType not in botConfigs:
            messagebox.showerror("Error", f"Unknown bot type: {botType}")
            return
        if not profileName.strip():
            messagebox.showerror("Error", "Profile name cannot be empty.")
            return
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]+", profileName.strip()):
            messagebox.showerror("Error", "Profile name may only contain letters, numbers, '_' and '-'.")
            return
        try:
            existing = set(self.controller.gameEnv.profileManager.listProfiles())
            if (self.profile is None or self.profile.name != profileName) and profileName in existing:
                messagebox.showerror("Error", f"A profile named '{profileName}' already exists.")
                return
        except Exception:
            pass

        for e in self.paramEntries.values():
            try:
                e.configure(style="TEntry")
            except Exception:
                pass
        for e in self.rewardEntries.values():
            try:
                e.configure(style="TEntry")
            except Exception:
                pass

        botParams: dict[str, float | None] = {}
        paramErrors: list[str] = []
        for paramKey, var in self.paramVars.items():
            txt = (var.get() or "").strip()
            if txt == "":
                botParams[paramKey] = None
                continue
            try:
                botParams[paramKey] = float(txt)
            except Exception:
                paramErrors.append(paramKey)

        rewardsConfig: dict[str, float] = {}
        rewardErrors: list[str] = []
        for rewardKey, var in self.rewardVars.items():
            txt = (var.get() or "").strip()
            try:
                rewardsConfig[rewardKey] = float(txt)
            except Exception:
                rewardErrors.append(rewardKey)

        if paramErrors or rewardErrors:
            for k in paramErrors:
                e = self.paramEntries.get(k)
                if e is not None:
                    try:
                        e.configure(style="Error.TEntry")
                    except Exception:
                        pass
            for k in rewardErrors:
                e = self.rewardEntries.get(k)
                if e is not None:
                    try:
                        e.configure(style="Error.TEntry")
                    except Exception:
                        pass

            def _format(keys: list[str], label: str) -> str:
                return (label + ":\n  - " + "\n  - ".join(keys)) if keys else ""

            msg = "\n\n".join(filter(None, [
                _format(paramErrors, "Invalid parameters"),
                _format(rewardErrors, "Invalid rewards"),
            ]))
            messagebox.showerror("Invalid Values", msg + "\n\nPlease fix these fields and try saving again.")
            return

        rawConfig = {k: v for k, v in botParams.items() if v is not None}
        botConfig = buildConfigForBotType(botType, rawConfig)

        rewardConfigObj = RewardConfig()
        rewardConfigObj.rewardModifiers.update(rewardsConfig)

        try:
            self.controller.gameEnv.setupNewProfile(profileName, botType, botConfig, rewardConfigObj)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profile: {e}")
            return

        try:
            self.controller.gameEnv.repository.ensureMazeFile(profileName)
        except Exception:
            pass

        messagebox.showinfo("Profile Saved", "Profile has been saved.")
        self.controller.frames["ProfileManagementFrame"].loadProfiles()
        self.controller.frames["VisualizationFrame"].loadProfiles()
        self.controller.frames["BotTrainingFrame"].loadProfiles()
        self.controller.showProfileManagement()

    def cancel(self) -> None:
        self.controller.showProfileManagement()
