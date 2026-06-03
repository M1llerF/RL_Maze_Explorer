# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import tkinter as tk
from tkinter import ttk
from typing import Any

from displayTools import DisplayTools
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
