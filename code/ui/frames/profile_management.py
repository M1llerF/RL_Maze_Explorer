import tkinter as tk
from tkinter import ttk, messagebox

from DisplayTools import DisplayTools


class ProfileManagementFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(self, text="Profile Management", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)
        ttk.Button(self, text="Create New Profile", command=self.create_new_profile).pack(pady=10)

        self.profile_list = tk.Listbox(self)
        self.profile_list.pack(pady=10)
        self.load_profiles()
        self.profile_list.bind("<Double-Button-1>", self.on_profile_double_click)

        ttk.Button(self, text="Delete Profile", command=self.delete_profile).pack(pady=10)

    def on_show(self):
        # Refresh when navigated back
        self.load_profiles()

    def load_profiles(self):
        DisplayTools.load_profiles(self.controller.game_env.profile_manager, self.profile_list)

    def create_new_profile(self):
        self.controller.show_create_edit_profile()

    def on_profile_double_click(self, event):
        selected_index = self.profile_list.curselection()
        if selected_index:
            profile_name = self.profile_list.get(selected_index)
            self.load_profile(profile_name)

    def load_profile(self, profile_name):
        profile = self.controller.game_env.profile_manager.load_profile(profile_name)
        self.controller.show_create_edit_profile(profile)

    def delete_profile(self):
        DisplayTools.delete_profile(self.controller.game_env.profile_manager, self.profile_list)
        self.load_profiles()
        self.controller.frames["BotTrainingFrame"].load_profiles()
        self.controller.frames["VisualizationFrame"].load_profiles()

