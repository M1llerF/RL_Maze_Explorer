# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false
import tkinter as tk
from tkinter import ttk
from typing import Any

from botConfigs import botConfigs, buildConfigForBotType
from rewardSystem import RewardConfig
from botProfile import BotProfile
from ui.eventBus import PROFILE_SAVED
from ui.scrollable import VerticalScrolledFrame


class HoverTooltip:
    def __init__(self, root: tk.Misc) -> None:
        self._root = root
        self._tipWindow: tk.Toplevel | None = None
        self._afterId: str | None = None
        self._activeWidget: tk.Widget | None = None

    def bind(self, widget: tk.Widget, text: str) -> None:
        if not text.strip():
            return
        widget.bind("<Enter>", lambda _event, w=widget, t=text: self._schedule(w, t), add="+")
        widget.bind("<Leave>", lambda _event: self.hide(), add="+")
        widget.bind("<ButtonPress>", lambda _event: self.hide(), add="+")

    def _schedule(self, widget: tk.Widget, text: str) -> None:
        self.hide()
        self._activeWidget = widget
        self._afterId = self._root.after(350, lambda: self._show(widget, text))

    def _show(self, widget: tk.Widget, text: str) -> None:
        self.hide()
        if not widget.winfo_exists():
            return
        self._activeWidget = widget
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.attributes("-topmost", True)
        label = tk.Label(
            tip,
            text=text,
            justify=tk.LEFT,
            bg="#fff8dc",
            fg="#222222",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            wraplength=320,
        )
        label.pack()
        x = widget.winfo_rootx() + widget.winfo_width() + 12
        y = widget.winfo_rooty() - 2
        tip.wm_geometry(f"+{x}+{y}")
        self._tipWindow = tip

    def hide(self) -> None:
        if self._afterId is not None:
            try:
                self._root.after_cancel(self._afterId)
            except Exception:
                pass
            self._afterId = None
        if self._tipWindow is not None:
            try:
                self._tipWindow.destroy()
            except Exception:
                pass
            self._tipWindow = None
        self._activeWidget = None


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
        self._tooltip = HoverTooltip(self)
        try:
            self._style = ttk.Style()
            self._style.configure("Error.TEntry", fieldbackground="#ffecec")
            self._style.configure("HelpIcon.TLabel", foreground="#1f5aa6")
        except Exception:
            self._style = None

        scrollHost = VerticalScrolledFrame(self)
        scrollHost.pack(fill=tk.BOTH, expand=True)
        content = scrollHost.content

        ttk.Label(content, text="Create/Edit Profile", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)

        buttonRow = ttk.Frame(content)
        buttonRow.pack(pady=6)
        ttk.Button(buttonRow, text="Save", command=self.saveProfile).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttonRow, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=6)
        self.statusVar = tk.StringVar(value="Fill in the profile details, then save.")
        self.statusLabel = tk.Label(
            content,
            textvariable=self.statusVar,
            fg="#805b00",
            justify=tk.LEFT,
            anchor="w",
            wraplength=920,
        )
        self.statusLabel.pack(fill=tk.X, padx=20, pady=(0, 8))

        ttk.Label(content, text="Profile Name:").pack()
        self.profileNameEntry = ttk.Entry(content)
        self.profileNameEntry.pack()

        ttk.Label(content, text="Bot Type:").pack()
        self.botTypeEntry = ttk.Combobox(content, values=list(botConfigs.keys()))
        self.botTypeEntry.pack()
        self.botTypeEntry.bind("<<ComboboxSelected>>", self.updateBotConfigUi)

        self.configFrame = ttk.Frame(content)
        self.configFrame.pack(fill="both", expand=True, pady=10)

    @staticmethod
    def _specTabs(config: dict[str, Any]) -> list[dict[str, str]]:
        rawTabs = config.get("tabs")
        tabs: list[dict[str, str]] = []
        if isinstance(rawTabs, list):
            for item in rawTabs:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("key", "")).strip()
                label = str(item.get("label", "")).strip()
                if key and label:
                    tabs.append({"key": key, "label": label})
        if tabs:
            return tabs
        return [{"key": "general", "label": "General"}, {"key": "rewards", "label": "Rewards"}]

    @staticmethod
    def _normalizePenaltyValue(value: float) -> float:
        return 0.0 if value == 0 else -abs(float(value))

    def _setStatus(self, message: str, *, tone: str = "info") -> None:
        colors = {
            "info": "#374151",
            "success": "#166534",
            "warning": "#805b00",
            "error": "#b91c1c",
        }
        self.statusVar.set(message)
        self.statusLabel.configure(fg=colors.get(tone, colors["info"]))

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
        paramHelp = dict(config.get("paramHelp", {})) if isinstance(config.get("paramHelp"), dict) else {}
        rewardHelp = dict(config.get("rewardHelp", {})) if isinstance(config.get("rewardHelp"), dict) else {}
        paramTabs = dict(config.get("paramTabs", {})) if isinstance(config.get("paramTabs"), dict) else {}
        rewardLabels = dict(config.get("rewardLabels", {})) if isinstance(config.get("rewardLabels"), dict) else {}

        notebook = ttk.Notebook(self.configFrame)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        tabs: dict[str, ttk.Frame] = {}
        for tabSpec in self._specTabs(config):
            frame = ttk.Frame(notebook)
            tabs[tabSpec["key"]] = frame
            notebook.add(frame, text=tabSpec["label"])
        self.currentConfigWidgets.extend([notebook, *tabs.values()])
        generalTab = next(iter(tabs.values()))
        paramRows: dict[str, tk.Widget] = {}
        rewardRows: dict[str, tk.Widget] = {}

        if "params" in config:
            for paramName, paramKey in config["params"].items():
                host = tabs.get(str(paramTabs.get(paramKey, "general")), generalTab)
                row = ttk.Frame(host)
                row.pack(fill="x", pady=2, padx=8)
                label = ttk.Label(row, text=f"{paramName}:")
                label.pack(side=tk.LEFT)
                helpText = str(paramHelp.get(paramKey, "")).strip()
                if helpText:
                    helpLabel = ttk.Label(row, text=" (?)", style="HelpIcon.TLabel", cursor="hand2")
                    helpLabel.pack(side=tk.LEFT)
                    self._tooltip.bind(helpLabel, helpText)
                    self._tooltip.bind(label, helpText)
                    self.currentConfigWidgets.append(helpLabel)
                var = tk.StringVar()
                entry = ttk.Entry(row, textvariable=var, width=14)
                entry.pack(side=tk.LEFT, padx=6)
                self.currentConfigWidgets.extend([row, label, entry])
                paramRows[str(paramKey)] = row
                self.paramVars[paramKey] = var
                self.paramEntries[paramKey] = entry

        rewardsTab = tabs.get("rewards")
        if "rewards" in config and rewardsTab is not None:
            for rewardKey, defaultValue in config["rewards"].items():
                row = ttk.Frame(rewardsTab)
                row.pack(fill="x", padx=8, pady=2)
                rewardLabelText = str(rewardLabels.get(rewardKey, rewardKey))
                rewardLabel = ttk.Label(row, text=f"{rewardLabelText}:")
                rewardLabel.pack(side=tk.LEFT)
                helpText = str(rewardHelp.get(rewardKey, "")).strip()
                if helpText:
                    helpLabel = ttk.Label(row, text=" (?)", style="HelpIcon.TLabel", cursor="hand2")
                    helpLabel.pack(side=tk.LEFT)
                    self._tooltip.bind(helpLabel, helpText)
                    self._tooltip.bind(rewardLabel, helpText)
                    self.currentConfigWidgets.append(helpLabel)
                var = tk.StringVar(value=defaultValue)
                rewardEntry = ttk.Entry(row, textvariable=var, width=14)
                rewardEntry.pack(side=tk.LEFT, padx=6)
                self.currentConfigWidgets.extend([row, rewardLabel, rewardEntry])
                rewardRows[str(rewardKey)] = row
                self.rewardVars[rewardKey] = var
                self.rewardEntries[rewardKey] = rewardEntry
            self._layoutRewardTab(rewardsTab, config, paramRows, rewardRows)

    def _layoutRewardTab(
        self,
        rewardsTab: tk.Widget,
        config: dict[str, Any],
        paramRows: dict[str, tk.Widget],
        rewardRows: dict[str, tk.Widget],
    ) -> None:
        sectionSpecs = config.get("rewardSections")
        if not isinstance(sectionSpecs, list):
            return

        groupedRows = dict(paramRows)
        groupedRows.update(rewardRows)

        seenKeys: set[str] = set()
        for sectionSpec in sectionSpecs:
            if not isinstance(sectionSpec, dict):
                continue
            title = str(sectionSpec.get("title", "")).strip()
            rawKeys = sectionSpec.get("keys", [])
            keys = [str(key) for key in rawKeys] if isinstance(rawKeys, list) else []
            if not title or not keys:
                continue
            header = ttk.Label(rewardsTab, text=title)
            header.pack(anchor="w", padx=8, pady=(10 if seenKeys else 2, 6))
            self.currentConfigWidgets.append(header)
            for key in keys:
                row = groupedRows.get(key)
                if row is None:
                    continue
                seenKeys.add(key)
                row.pack_forget()
                row.pack(fill="x", padx=8, pady=2)

    # ----- Data binding -----
    def loadProfile(self, profile: Any = None) -> None:
        self.profile = profile
        self.profileNameEntry.delete(0, tk.END)
        self.botTypeEntry.set("")
        self.updateBotConfigUi()
        self._setStatus("Fill in the profile details, then save.", tone="info")
        if not profile:
            return

        self.profileNameEntry.insert(0, profile.name)
        self.botTypeEntry.set(profile.botType)
        self.updateBotConfigUi()

        if profile.config:
            for paramKey, var in self.paramVars.items():
                value = getattr(profile.config, paramKey, "")
                if paramKey == "macroOptionSet":
                    if value == "naive":
                        value = 1
                    elif value == "momentum":
                        value = 2
                    elif value == "astar":
                        value = 3
                var.set("" if value is None else value)

        if profile.rewardConfig:
            for rewardKey, var in self.rewardVars.items():
                var.set(profile.rewardConfig.rewardModifiers.get(rewardKey, ""))

    # ----- Actions -----
    def saveProfile(self) -> None:
        profileName = self.profileNameEntry.get()
        botType = self.botTypeEntry.get()

        if botType not in botConfigs:
            self._setStatus(f"Unknown bot type: {botType}", tone="error")
            return
        if not profileName.strip():
            self._setStatus("Profile name cannot be empty.", tone="error")
            return
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]+", profileName.strip()):
            self._setStatus("Profile name may only contain letters, numbers, '_' and '-'.", tone="error")
            return
        try:
            existing = set(self.controller.gameEnv.profileManager.listProfiles())
            if (self.profile is None or self.profile.name != profileName) and profileName in existing:
                self._setStatus(f"A profile named '{profileName}' already exists.", tone="error")
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
                if paramKey == "macroOptionSet":
                    botParams[paramKey] = int(float(txt))
                else:
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
            self._setStatus(msg + "\n\nPlease fix these fields and try saving again.", tone="error")
            return

        rewardClipKeys = ("rewardClipMin", "rewardClipMax")
        clipValues = [botParams.get(key) for key in rewardClipKeys]
        if any(value is None for value in clipValues) and any(value is not None for value in clipValues):
            self._setStatus(
                "Reward Clip Min and Reward Clip Max must both be filled in or both left blank.",
                tone="error",
            )
            return

        negativeParams = set(botConfigs.get(botType, {}).get("negativeParams", []))
        negativeRewards = set(botConfigs.get(botType, {}).get("negativeRewards", []))
        for key in negativeParams:
            value = botParams.get(key)
            if value is not None:
                botParams[key] = self._normalizePenaltyValue(value)
        for key in negativeRewards:
            if key in rewardsConfig:
                rewardsConfig[key] = self._normalizePenaltyValue(rewardsConfig[key])

        rawConfig = {k: v for k, v in botParams.items() if v is not None}
        try:
            botConfig = buildConfigForBotType(botType, rawConfig)
        except ValueError as e:
            self._setStatus(str(e), tone="error")
            return

        rewardConfigObj = RewardConfig()
        rewardConfigObj.rewardModifiers.update({k: str(v) for k, v in rewardsConfig.items()})

        try:
            previousName = self.profile.name if self.profile is not None else None
            self.controller.gameEnv.setupNewProfile(
                profileName,
                botType,
                botConfig,
                rewardConfigObj,
                previousName=previousName,
            )
        except Exception as e:
            self._setStatus(f"Failed to save profile: {e}", tone="error")
            return

        self.controller.eventBus.emit(PROFILE_SAVED, profileName=profileName)
        self.controller.showProfileManagement()

    def cancel(self) -> None:
        self.controller.showProfileManagement()
