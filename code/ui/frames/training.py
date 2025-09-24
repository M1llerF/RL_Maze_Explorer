import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


class BotTrainingFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.visualization_window = None
        self.training_active = False
        self.last_selected_profile = ""
        self._log_interval = 1
        self._log_last_round = 0

        ttk.Label(self, text="Bot Training", font=("TkDefaultFont", 20)).pack(pady=10, padx=10)

        ttk.Label(self, text="Select Profile:").pack()
        self.profile_select = ttk.Combobox(self, state="readonly")
        self.profile_select.pack()

        ttk.Label(self, text="Number of Rounds:").pack()
        self.rounds_entry = ttk.Entry(self)
        self.rounds_entry.pack()

        ttk.Label(self, text="Maze Source:").pack(pady=(10, 0))
        self.maze_mode = ttk.Combobox(self, values=["Random", "Fixed (Builder)", "Pool"], state="readonly")
        self.maze_mode.set("Random")
        self.maze_mode.pack()

        def on_maze_mode_change(event=None):
            mode = self.maze_mode.get()
            if mode == "Fixed (Builder)":
                if not self.controller.game_env.fixed_maze_active or self.controller.game_env.fixed_maze_state is None:
                    path = filedialog.askopenfilename(
                        title="Choose a maze JSON to use as Fixed",
                        initialdir='mazes',
                        filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
                    )
                    if path:
                        try:
                            with open(path, 'r') as f:
                                state = json.load(f)
                            self.controller.game_env.set_fixed_maze(state)
                            messagebox.showinfo("Fixed Maze Set", f"Using {os.path.basename(path)} for training.")
                        except Exception as e:
                            messagebox.showerror("Invalid Maze", f"Could not load maze: {e}")
                            self.maze_mode.set("Random")
                    else:
                        self.maze_mode.set("Random")

        self.maze_mode.bind("<<ComboboxSelected>>", on_maze_mode_change)

        self.open_builder_btn = ttk.Button(self, text="Open Maze Builder", command=lambda: self.controller.show_maze_builder())
        self.open_builder_btn.pack(pady=6)

        actions = ttk.Frame(self)
        actions.pack(pady=10)
        self.start_btn = ttk.Button(actions, text="Start Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=6)
        self.stop_btn = ttk.Button(actions, text="Stop", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=6)
        self.training_progress = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self.training_progress.pack(pady=10)
        self.log_output = tk.Text(self, height=10, width=50)
        self.log_output.pack(pady=10)

        ttk.Button(self, text="Open Visualization", command=self.open_visualization).pack(pady=10)
        self.status_hint = tk.Label(self, text="", fg="#805b00")
        self.status_hint.pack(pady=(0, 6))

        self.load_profiles()

    def on_show(self):
        self.load_profiles()

    def load_profiles(self):
        profiles = self.controller.game_env.profile_manager.list_profiles()
        self.profile_select['values'] = profiles
        if self.last_selected_profile and self.last_selected_profile in profiles:
            try:
                self.profile_select.set(self.last_selected_profile)
            except Exception:
                pass

    def start_training(self):
        if self.training_active:
            return
        selected_profile = self.profile_select.get()
        if not selected_profile:
            messagebox.showerror("Error", "No profile selected.")
            return

        rounds_txt = self.rounds_entry.get()
        if not rounds_txt.isdigit():
            messagebox.showerror("Error", "Number of rounds must be a positive integer.")
            return
        rounds = int(rounds_txt)

        mode = self.maze_mode.get() or "Random"
        if mode == "Fixed (Builder)":
            if not self.controller.game_env.fixed_maze_active or self.controller.game_env.fixed_maze_state is None:
                messagebox.showerror("No Fixed Maze", "No fixed maze set. Open Maze Builder to create or load one, then click 'Use In Training'.")
                return

        self.training_progress['maximum'] = rounds
        self.training_progress['value'] = 0
        self._log_interval = max(1, rounds // 100)
        self._log_last_round = 0
        self.log_output.delete("1.0", tk.END)
        self.log_output.insert(tk.END, f"Training started for {selected_profile} with {rounds} rounds...\n")
        self.last_selected_profile = selected_profile
        self.set_controls_enabled(False)
        self.training_active = True
        try:
            self.stop_btn.configure(state="normal")
        except Exception:
            pass

        def on_progress(done: int, total: int):
            try:
                self.controller.root.after(0, self.update_progress, done, total)
            except Exception:
                pass

        def on_error(err: Exception):
            def _report():
                try:
                    self.log_output.insert(tk.END, f"Training error: {err}\n")
                    self.log_output.see(tk.END)
                except Exception:
                    pass
                self.training_active = False
                self.set_controls_enabled(True)
                try:
                    self.stop_btn.configure(state="disabled")
                except Exception:
                    pass
            try:
                self.controller.root.after(0, _report)
            except Exception:
                pass

        def on_complete():
            def _done():
                self.training_active = False
                self.set_controls_enabled(True)
                try:
                    self.stop_btn.configure(state="disabled")
                except Exception:
                    pass
            try:
                self.controller.root.after(0, _done)
            except Exception:
                pass

        pool_size = max(5, min(25, rounds // 10 or 5)) if mode == "Pool" else 0
        self.controller.training_controller.start(
            profile_name=selected_profile,
            rounds=rounds,
            maze_mode=mode,
            pool_size=pool_size or 20,
            on_progress=on_progress,
            on_error=on_error,
            on_complete=on_complete,
        )

    def set_controls_enabled(self, enabled: bool):
        state = "readonly" if enabled else "disabled"
        entry_state = "normal" if enabled else "disabled"
        try:
            self.profile_select.configure(state=state)
        except Exception:
            pass
        try:
            self.maze_mode.configure(state=state)
        except Exception:
            pass
        try:
            self.rounds_entry.configure(state=entry_state)
        except Exception:
            pass
        try:
            self.open_builder_btn.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            pass

    def stop_training(self):
        if not self.training_active:
            return
        try:
            self.stop_btn.configure(state="disabled")
        except Exception:
            pass
        try:
            self.controller.training_controller.stop()
        except Exception:
            pass

    def update_progress(self, completed_rounds, total_rounds):
        self.training_progress['value'] = completed_rounds
        if (
            completed_rounds == total_rounds
            or completed_rounds == 1
            or completed_rounds - self._log_last_round >= self._log_interval
        ):
            self.log_output.insert(tk.END, f"Completed round {completed_rounds}/{total_rounds}\n")
            self.log_output.see(tk.END)
            self._log_last_round = completed_rounds
        if completed_rounds == total_rounds:
            self.log_output.insert(tk.END, "Training completed.\n")
            self.log_output.see(tk.END)
            self.training_active = False
            self.set_controls_enabled(True)

    def cancel_training_poll(self):
        return

    def open_visualization(self):
        from ui.frames.visualization import VisualizationWindow

        selected_profile = self.profile_select.get()
        if not selected_profile:
            messagebox.showerror("Error", "No profile selected.")
            return
        profile = self.controller.game_env.profile_manager.load_profile(selected_profile)
        profile_index = self.controller.game_env.apply_profile(profile)

        if getattr(self, 'visualization_window', None) and self.visualization_window.winfo_exists():
            self.visualization_window.focus()
        else:
            self.visualization_window = VisualizationWindow(self.controller.root, self.controller.game_env, selected_profile, profile_index)
            try:
                self.log_output.insert(tk.END, "Opened visualization. Training is PAUSED while the window is open.\n")
                self.log_output.see(tk.END)
                self.status_hint.configure(text="Note: Training is paused while visualization is open.")
            except Exception:
                pass

