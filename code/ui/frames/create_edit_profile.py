import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

from BotConfigs import bot_configs, QLearningConfig
from RewardSystem import RewardConfig
from BotStatistics import BotStatistics
from BotProfile import BotProfile


class CreateEditProfileFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.current_config_widgets = []
        self.param_vars = {}
        self.reward_vars = {}
        self.auto_vars = {}
        self.param_entries = {}
        self.reward_entries = {}
        try:
            self._style = ttk.Style()
            self._style.configure("Error.TEntry", fieldbackground="#ffecec")
        except Exception:
            self._style = None

        ttk.Label(self, text="Create/Edit Profile", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)

        ttk.Label(self, text="Profile Name:").pack()
        self.profile_name_entry = ttk.Entry(self)
        self.profile_name_entry.pack()

        ttk.Label(self, text="Bot Type:").pack()
        self.bot_type_entry = ttk.Combobox(self, values=list(bot_configs.keys()))
        self.bot_type_entry.pack()
        self.bot_type_entry.bind("<<ComboboxSelected>>", self.update_bot_config_ui)

        self.config_frame = ttk.Frame(self)
        self.config_frame.pack(pady=10)

        ttk.Button(self, text="Save", command=self.save_profile).pack(pady=10)
        ttk.Button(self, text="Cancel", command=self.cancel).pack(pady=10)

    # ----- UI building -----
    def update_bot_config_ui(self, event=None):
        for widget in self.current_config_widgets:
            widget.destroy()
        self.current_config_widgets.clear()
        self.param_vars.clear()
        self.reward_vars.clear()
        self.auto_vars.clear()
        self.param_entries.clear()

        bot_type = self.bot_type_entry.get()
        if bot_type not in bot_configs:
            return
        config = bot_configs[bot_type]

        # Parameters
        if "params" in config:
            for param_name, param_key in config["params"].items():
                row = ttk.Frame(self.config_frame)
                row.pack(fill="x", pady=2)
                label = ttk.Label(row, text=f"{param_name}:")
                label.pack(side=tk.LEFT)
                var = tk.StringVar()
                entry = ttk.Entry(row, textvariable=var, width=12)
                entry.pack(side=tk.LEFT, padx=6)
                self.current_config_widgets.extend([row, label, entry])
                self.param_vars[param_key] = var
                self.param_entries[param_key] = entry

        # Rewards
        if "rewards" in config:
            label = ttk.Label(self.config_frame, text="Reward Configuration:")
            label.pack()
            self.current_config_widgets.append(label)
            for reward_key, default_value in config["rewards"].items():
                reward_label = ttk.Label(self.config_frame, text=reward_key)
                reward_label.pack()
                var = tk.StringVar(value=default_value)
                reward_entry = ttk.Entry(self.config_frame, textvariable=var)
                reward_entry.pack()
                self.current_config_widgets.extend([reward_label, reward_entry])
                self.reward_vars[reward_key] = var
                self.reward_entries[reward_key] = reward_entry

    # ----- Data binding -----
    def load_profile(self, profile=None):
        self.profile = profile
        if profile:
            self.profile_name_entry.delete(0, tk.END)
            self.profile_name_entry.insert(0, profile.name)
            self.bot_type_entry.set(profile.bot_type)
            self.update_bot_config_ui()

            if profile.config:
                for param_key, var in self.param_vars.items():
                    var.set(getattr(profile.config, param_key, ""))
            # No algorithm-specific auto flags

            if profile.reward_config:
                for reward_key, var in self.reward_vars.items():
                    var.set(profile.reward_config.reward_modifiers.get(reward_key, ""))

    # ----- Actions -----
    def save_profile(self):
        profile_name = self.profile_name_entry.get()
        bot_type = self.bot_type_entry.get()

        if bot_type not in bot_configs:
            messagebox.showerror("Error", f"Unknown bot type: {bot_type}")
            return
        if not profile_name.strip():
            messagebox.showerror("Error", "Profile name cannot be empty.")
            return
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]+", profile_name.strip()):
            messagebox.showerror("Error", "Profile name may only contain letters, numbers, '_' and '-'.")
            return
        try:
            existing = set(self.controller.game_env.profile_manager.list_profiles())
            if (self.profile is None or self.profile.name != profile_name) and profile_name in existing:
                messagebox.showerror("Error", f"A profile named '{profile_name}' already exists.")
                return
        except Exception:
            pass

        for e in self.param_entries.values():
            try:
                e.configure(style="TEntry")
            except Exception:
                pass
        for e in self.reward_entries.values():
            try:
                e.configure(style="TEntry")
            except Exception:
                pass

        config = bot_configs[bot_type]
        bot_params = {}
        param_errors = []
        for param_key, var in self.param_vars.items():
            txt = (var.get() or "").strip()
            if txt == "":
                bot_params[param_key] = None
                continue
            try:
                bot_params[param_key] = float(txt)
            except Exception:
                param_errors.append(param_key)

        rewards_config = {}
        reward_errors = []
        for reward_key, var in self.reward_vars.items():
            txt = (var.get() or "").strip()
            try:
                rewards_config[reward_key] = float(txt)
            except Exception:
                reward_errors.append(reward_key)

        if param_errors or reward_errors:
            for k in param_errors:
                e = self.param_entries.get(k)
                if e is not None:
                    try:
                        e.configure(style="Error.TEntry")
                    except Exception:
                        pass
            for k in reward_errors:
                e = self.reward_entries.get(k)
                if e is not None:
                    try:
                        e.configure(style="Error.TEntry")
                    except Exception:
                        pass
            def _format(keys, label):
                return (label + ":\n  - " + "\n  - ".join(keys)) if keys else ""
            msg = "\n\n".join(filter(None, [
                _format(param_errors, "Invalid parameters"),
                _format(reward_errors, "Invalid rewards"),
            ]))
            messagebox.showerror("Invalid Values", msg + "\n\nPlease fix these fields and try saving again.")
            return

        if bot_type == "QLearningBot":
            q_defaults = QLearningConfig()
            bot_config = QLearningConfig(
                learning_rate=(bot_params.get('learning_rate') if bot_params.get('learning_rate') is not None else q_defaults.learning_rate),
                discount_factor=(bot_params.get('discount_factor') if bot_params.get('discount_factor') is not None else q_defaults.discount_factor),
                use_position_in_state=bool(getattr(q_defaults, 'use_position_in_state', True))
            )
        else:
            bot_config = None

        reward_config_obj = RewardConfig()
        reward_config_obj.reward_modifiers.update(rewards_config)

        bot_specific_data = {}

        profile = BotProfile(
            name=profile_name,
            bot_type=bot_type,
            config=bot_config,
            reward_config=reward_config_obj,
            statistics=BotStatistics(),
            bot_specific_data=bot_specific_data
        )

        try:
            self.controller.game_env.setup_new_profile(profile_name, bot_type, bot_config, reward_config_obj)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profile: {e}")
            return

        try:
            # Initialize default mazes.json via repository
            self.controller.game_env.repository.ensure_maze_file(profile_name)
        except Exception:
            pass

        messagebox.showinfo("Profile Saved", "Profile has been saved.")

        self.controller.frames["ProfileManagementFrame"].load_profiles()
        self.controller.frames["VisualizationFrame"].load_profiles()
        self.controller.frames["BotTrainingFrame"].load_profiles()
        self.controller.show_profile_management()

    def cancel(self):
        self.controller.show_profile_management()
