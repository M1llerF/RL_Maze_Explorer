# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import tkinter as tk
from tkinter import ttk
from typing import Any

from displayTools import DisplayTools


class ProfileManagementFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller

        ttk.Label(self, text="Profile Management", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)
        ttk.Button(self, text="Create New Profile", command=self.createNewProfile).pack(pady=10)

        self.profileList = tk.Listbox(self)
        self.profileList.pack(pady=10)
        self.loadProfiles()
        self.profileList.bind("<Double-Button-1>", self.onProfileDoubleClick)

        ttk.Button(self, text="Delete Profile", command=self.deleteProfile).pack(pady=10)

    def onShow(self) -> None:
        # Refresh when navigated back
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
        DisplayTools.deleteProfile(self.controller.gameEnv.profileManager, self.profileList)
        self.loadProfiles()
        self.controller.frames["BotTrainingFrame"].loadProfiles()
        self.controller.frames["VisualizationFrame"].loadProfiles()

