import os
import tkinter as tk

from GameEnvironment import GameEnvironment
from services.training import TrainingController


class MazeAIApp:
    """Main application wiring for Maze AI.

    Responsibilities:
    - Own shared services (GameEnvironment, TrainingController)
    - Build navigation and host frame instances
    - Provide navigation helpers used by frames
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Maze AI Experiment")
        # Ensure a default folder exists for user mazes
        try:
            os.makedirs('mazes', exist_ok=True)
            os.makedirs('profiles', exist_ok=True)
        except Exception:
            pass

        self.game_env = GameEnvironment()
        self.training_controller = TrainingController(self.game_env)

        self.create_navigation_bar()
        self.create_main_frames()
        # Graceful shutdown handler to cancel pending timers
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----- UI building -----
    def create_navigation_bar(self):
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        self.nav_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Navigation", menu=self.nav_menu)
        self.nav_menu.add_command(label="Profile Management", command=self.show_profile_management)
        self.nav_menu.add_command(label="Bot Training", command=self.show_bot_training)
        self.nav_menu.add_command(label="Maze Builder", command=self.show_maze_builder)
        self.nav_menu.add_command(label="Visualizations", command=self.show_visualizations)
        self.nav_menu.add_command(label="Exit", command=self.root.quit)

    def create_main_frames(self):
        # Import frames locally to avoid circular imports at module load
        from ui.frames.profile_management import ProfileManagementFrame
        from ui.frames.create_edit_profile import CreateEditProfileFrame
        from ui.frames.training import BotTrainingFrame
        from ui.frames.maze_builder import MazeBuilderFrame
        from ui.frames.visualization import VisualizationFrame

        self.frames = {}
        for F in (
            ProfileManagementFrame,
            CreateEditProfileFrame,
            BotTrainingFrame,
            MazeBuilderFrame,
            VisualizationFrame,
        ):
            page_name = F.__name__
            frame = F(parent=self.root, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Default screen
        self.show_frame("ProfileManagementFrame")

    # ----- Navigation helpers -----
    def show_frame(self, page_name: str):
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, 'on_show'):
            try:
                frame.on_show()
            except Exception:
                pass

    def show_profile_management(self):
        self.show_frame("ProfileManagementFrame")

    def show_create_edit_profile(self, profile=None):
        frame = self.frames["CreateEditProfileFrame"]
        frame.load_profile(profile)
        self.show_frame("CreateEditProfileFrame")

    def show_bot_training(self):
        self.show_frame("BotTrainingFrame")

    def show_visualizations(self):
        self.show_frame("VisualizationFrame")

    def show_maze_builder(self):
        self.show_frame("MazeBuilderFrame")

    def on_close(self):
        # Attempt to cancel any legacy polling before destroying root
        try:
            bt = self.frames.get("BotTrainingFrame")
            if bt and hasattr(bt, 'cancel_training_poll'):
                bt.cancel_training_poll()
        except Exception:
            pass
        # Request training controller to stop background thread
        try:
            self.training_controller.stop()
        except Exception:
            pass
        # Proactively destroy Matplotlib Tk widgets to avoid Tk finalizer errors
        try:
            viz = self.frames.get("VisualizationFrame")
            if viz is not None and getattr(viz, 'canvas_agg', None):
                try:
                    widget = viz.canvas_agg.get_tk_widget()
                    viz.canvas_agg = None
                    widget.destroy()
                except Exception:
                    pass
            # Close any remaining Matplotlib figures
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception:
                pass
        except Exception:
            pass
        self.root.destroy()
