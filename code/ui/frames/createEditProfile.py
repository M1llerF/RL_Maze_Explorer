# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, cast

from botConfigs import botConfigs, QLearningConfig
from rewardSystem import RewardConfig
from botProfile import BotProfile


class CreateEditProfileFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller
        self.currentConfigWidgets = []
        self.paramVars = {}
        self.rewardVars = {}
        self.autoVars = {}
        self.paramEntries = {}
        self.rewardEntries = {}
        self.profile: BotProfile | None = None
        try:
            self._style = ttk.Style()
            self._style.configure("Error.TEntry", fieldbackground="#ffecec")
        except Exception:
            self._style = None

        ttk.Label(self, text="Create/Edit Profile", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)

        ttk.Label(self, text="Profile Name:").pack()
        self.profileNameEntry = ttk.Entry(self)
        self.profileNameEntry.pack()

        ttk.Label(self, text="Bot Type:").pack()
        self.botTypeEntry = ttk.Combobox(self, values=list(botConfigs.keys()))
        self.botTypeEntry.pack()
        self.botTypeEntry.bind("<<ComboboxSelected>>", self.updateBotConfigUi)

        self.configFrame = ttk.Frame(self)
        self.configFrame.pack(pady=10)

        ttk.Button(self, text="Save", command=self.saveProfile).pack(pady=10)
        ttk.Button(self, text="Cancel", command=self.cancel).pack(pady=10)

    # ----- UI building -----
    def updateBotConfigUi(self, event: Any = None) -> None:
        for widget in self.currentConfigWidgets:
            widget.destroy()
        self.currentConfigWidgets.clear()
        self.paramVars.clear()
        self.rewardVars.clear()
        self.autoVars.clear()
        self.paramEntries.clear()

        botType = self.botTypeEntry.get()
        if botType not in botConfigs:
            return
        config = botConfigs[botType]

        # Parameters
        if "params" in config:
            for paramName, paramKey in config["params"].items():
                row = ttk.Frame(self.configFrame)
                row.pack(fill="x", pady=2)
                label = ttk.Label(row, text=f"{paramName}:")
                label.pack(side=tk.LEFT)
                var = tk.StringVar()
                entry = ttk.Entry(row, textvariable=var, width=12)
                entry.pack(side=tk.LEFT, padx=6)
                self.currentConfigWidgets.extend([row, label, entry])
                self.paramVars[paramKey] = var
                self.paramEntries[paramKey] = entry

        # Rewards
        if "rewards" in config:
            label = ttk.Label(self.configFrame, text="Reward Configuration:")
            label.pack()
            self.currentConfigWidgets.append(label)
            for rewardKey, defaultValue in config["rewards"].items():
                rewardLabel = ttk.Label(self.configFrame, text=rewardKey)
                rewardLabel.pack()
                var = tk.StringVar(value=defaultValue)
                rewardEntry = ttk.Entry(self.configFrame, textvariable=var)
                rewardEntry.pack()
                self.currentConfigWidgets.extend([rewardLabel, rewardEntry])
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
            # No algorithm-specific auto flags

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

        botParams = {}
        paramErrors = []
        for paramKey, var in self.paramVars.items():
            txt = (var.get() or "").strip()
            if txt == "":
                botParams[paramKey] = None
                continue
            try:
                botParams[paramKey] = float(txt)
            except Exception:
                paramErrors.append(paramKey)

        rewardsConfig = {}
        rewardErrors = []
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
            def _format(keys, label):
                return (label + ":\n  - " + "\n  - ".join(keys)) if keys else ""
            msg = "\n\n".join(filter(None, [
                _format(paramErrors, "Invalid parameters"),
                _format(rewardErrors, "Invalid rewards"),
            ]))
            messagebox.showerror("Invalid Values", msg + "\n\nPlease fix these fields and try saving again.")
            return

        if botType == "QLearningBot":
            qDefaults = QLearningConfig()
            botConfig = QLearningConfig(
                learningRate=cast(float, botParams.get('learning_rate') if botParams.get('learning_rate') is not None else qDefaults.learningRate),
                discountFactor=cast(float, botParams.get('discount_factor') if botParams.get('discount_factor') is not None else qDefaults.discountFactor),
                usePositionInState=bool(getattr(qDefaults, 'use_position_in_state', True))
            )
        else:
            botConfig = QLearningConfig()

        rewardConfigObj = RewardConfig()
        rewardConfigObj.rewardModifiers.update(rewardsConfig)

        try:
            self.controller.gameEnv.setupNewProfile(profileName, botType, botConfig, rewardConfigObj)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profile: {e}")
            return

        try:
            # Initialize default mazes.json via repository
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
