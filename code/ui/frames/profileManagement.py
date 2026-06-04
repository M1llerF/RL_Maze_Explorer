# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any

from displayTools import DisplayTools
from ui.event_bus import PROFILE_SAVED, PROFILE_DELETED
from ui.scrollable import VerticalScrolledFrame


class ProfileManagementFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller

        scrollHost = VerticalScrolledFrame(self)
        scrollHost.pack(fill=tk.BOTH, expand=True)
        content = scrollHost.content

        ttk.Label(content, text="Profile Management", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)
        ttk.Button(content, text="Create New Profile", command=self.createNewProfile).pack(pady=10)

        listFrame = ttk.Frame(content)
        listFrame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.profileList = tk.Listbox(listFrame)
        listScrollbar = ttk.Scrollbar(listFrame, orient="vertical", command=self.profileList.yview)
        self.profileList.configure(yscrollcommand=listScrollbar.set)
        self.profileList.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        listScrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.loadProfiles()
        self.profileList.bind("<Double-Button-1>", self.onProfileDoubleClick)

        ttk.Button(content, text="Delete Profile", command=self.deleteProfile).pack(pady=10)

        def _reloadProfiles(**_: Any) -> None:
            self.loadProfiles()
        controller.eventBus.subscribe(PROFILE_SAVED, _reloadProfiles)
        controller.eventBus.subscribe(PROFILE_DELETED, _reloadProfiles)

    def on_show(self) -> None:
        self.loadProfiles()

    def loadProfiles(self) -> None:
        DisplayTools.loadProfiles(self.controller.gameEnv.profileManager, self.profileList)

    def createNewProfile(self) -> None:
        self.controller.showCreateEditProfile()

    def onProfileDoubleClick(self, event: Any) -> None:
        selectedIndex = self.profileList.curselection()
        if selectedIndex:
            profileName = str(self.profileList.get(selectedIndex[0]))
            self.loadProfile(profileName)

    def loadProfile(self, profileName: str) -> None:
        profile = self.controller.gameEnv.profileManager.loadProfile(profileName)
        self.controller.showCreateEditProfile(profile)

    def deleteProfile(self) -> None:
        selectedIndex = self.profileList.curselection()
        if not selectedIndex:
            messagebox.showerror("Error", "No profile selected.")
            return
        profileName = str(self.profileList.get(selectedIndex[0]))
        try:
            self.controller.gameEnv.profileService.deleteProfile(profileName)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete profile '{profileName}'. Error: {e}")
            return
        self.profileList.delete(selectedIndex)
        self.controller.eventBus.emit(PROFILE_DELETED, profileName=profileName)
