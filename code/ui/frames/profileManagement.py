# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import tkinter as tk
from tkinter import ttk
from typing import Any

from displayTools import DisplayTools
from ui.eventBus import PROFILE_SAVED, PROFILE_DELETED
from ui.scrollable import VerticalScrolledFrame


class ProfileManagementFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller
        self._pendingDeleteProfileName: str | None = None

        scrollHost = VerticalScrolledFrame(self)
        scrollHost.pack(fill=tk.BOTH, expand=True)
        content = scrollHost.content

        ttk.Label(content, text="Profile Management", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)
        ttk.Button(content, text="Create New Profile", command=self.createNewProfile).pack(pady=10)
        self.statusVar = tk.StringVar(value="Select a profile to edit or delete.")
        self.statusLabel = tk.Label(
            content,
            textvariable=self.statusVar,
            fg="#805b00",
            justify=tk.LEFT,
            anchor="w",
            wraplength=900,
        )
        self.statusLabel.pack(fill=tk.X, padx=20, pady=(0, 8))

        listFrame = ttk.Frame(content)
        listFrame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.profileList = tk.Listbox(listFrame)
        listScrollbar = ttk.Scrollbar(listFrame, orient="vertical", command=self.profileList.yview)
        self.profileList.configure(yscrollcommand=listScrollbar.set)
        self.profileList.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        listScrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.loadProfiles()
        self.profileList.bind("<Double-Button-1>", self.onProfileDoubleClick)
        self.profileList.bind("<<ListboxSelect>>", self._onProfileSelectionChanged)

        self.deleteProfileBtn = ttk.Button(content, text="Delete Profile", command=self.deleteProfile)
        self.deleteProfileBtn.pack(pady=10)

        def _reloadProfiles(**_: Any) -> None:
            self.loadProfiles()
        def _handleProfileSaved(*, profileName: str = "", **_: Any) -> None:
            label = profileName or "profile"
            self._setStatus(f"Saved {label}.", tone="success")
        def _handleProfileDeleted(*, profileName: str = "", **_: Any) -> None:
            label = profileName or "profile"
            self._setStatus(f"Deleted {label}.", tone="success")
        controller.eventBus.subscribe(PROFILE_SAVED, _reloadProfiles)
        controller.eventBus.subscribe(PROFILE_DELETED, _reloadProfiles)
        controller.eventBus.subscribe(PROFILE_SAVED, _handleProfileSaved)
        controller.eventBus.subscribe(PROFILE_DELETED, _handleProfileDeleted)

    def on_show(self) -> None:
        self.loadProfiles()

    def loadProfiles(self) -> None:
        self._resetDeleteConfirmation()
        DisplayTools.loadProfiles(self.controller.gameEnv.profileManager, self.profileList)

    def createNewProfile(self) -> None:
        self._resetDeleteConfirmation()
        self.controller.showCreateEditProfile()

    def onProfileDoubleClick(self, event: Any) -> None:
        selectedIndex = self.profileList.curselection()
        if selectedIndex:
            profileName = str(self.profileList.get(selectedIndex[0]))
            self.loadProfile(profileName)

    def loadProfile(self, profileName: str) -> None:
        self._resetDeleteConfirmation()
        profile = self.controller.gameEnv.profileManager.loadProfile(profileName)
        self.controller.showCreateEditProfile(profile)

    def _onProfileSelectionChanged(self, event: Any = None) -> None:
        del event
        pending = self._pendingDeleteProfileName
        selected = self._selectedProfileName()
        if pending and pending != selected:
            self._resetDeleteConfirmation()

    def _selectedProfileName(self) -> str:
        selectedIndex = self.profileList.curselection()
        if not selectedIndex:
            return ""
        return str(self.profileList.get(selectedIndex[0]))

    def _setStatus(self, message: str, *, tone: str = "info") -> None:
        colors = {
            "info": "#374151",
            "success": "#166534",
            "warning": "#805b00",
            "error": "#b91c1c",
        }
        self.statusVar.set(message)
        self.statusLabel.configure(fg=colors.get(tone, colors["info"]))

    def _resetDeleteConfirmation(self) -> None:
        self._pendingDeleteProfileName = None
        try:
            self.deleteProfileBtn.configure(text="Delete Profile")
        except Exception:
            pass

    def deleteProfile(self) -> None:
        profileName = self._selectedProfileName()
        if not profileName:
            self._setStatus("Select a profile before deleting it.", tone="error")
            return
        if self._pendingDeleteProfileName != profileName:
            self._pendingDeleteProfileName = profileName
            self.deleteProfileBtn.configure(text="Confirm Delete")
            self._setStatus(
                f"Click Delete Profile again to permanently remove '{profileName}'.",
                tone="warning",
            )
            return
        try:
            self.controller.gameEnv.profileService.deleteProfile(profileName)
        except Exception as e:
            self._resetDeleteConfirmation()
            self._setStatus(f"Failed to delete '{profileName}': {e}", tone="error")
            return
        self._resetDeleteConfirmation()
        self.controller.eventBus.emit(PROFILE_DELETED, profileName=profileName)
