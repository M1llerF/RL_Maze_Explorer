import os
import tkinter as tk
from typing import Any

from gameEnvironment import GameEnvironment
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
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        # Ensure a default folder exists for user mazes
        try:
            os.makedirs('mazes', exist_ok=True)
            os.makedirs('profiles', exist_ok=True)
        except Exception:
            pass

        self.gameEnv = GameEnvironment()
        self.trainingController = TrainingController(self.gameEnv)

        self.createNavigationBar()
        self.createMainFrames()
        # Graceful shutdown handler to cancel pending timers
        self.root.protocol("WM_DELETE_WINDOW", self.onClose)

    # ----- UI building -----
    def createNavigationBar(self):
        self.menuBar = tk.Menu(self.root)
        self.root.config(menu=self.menuBar)

        self.navMenu = tk.Menu(self.menuBar, tearoff=0)
        self.menuBar.add_cascade(label="Navigation", menu=self.navMenu)
        self.navMenu.add_command(label="Profile Management", command=self.showProfileManagement)
        self.navMenu.add_command(label="Bot Training", command=self.showBotTraining)
        self.navMenu.add_command(label="Maze Builder", command=self.showMazeBuilder)
        self.navMenu.add_command(label="Visualizations", command=self.showVisualizations)
        self.navMenu.add_command(label="Exit", command=self.root.quit)

    def createMainFrames(self):
        # Import frames locally to avoid circular imports at module load
        from ui.frames.profileManagement import ProfileManagementFrame
        from ui.frames.createEditProfile import CreateEditProfileFrame
        from ui.frames.training import BotTrainingFrame
        from ui.frames.mazeBuilder import MazeBuilderFrame
        from ui.frames.visualization import VisualizationFrame

        self.frames: dict[str, Any] = {}
        for F in (
            ProfileManagementFrame,
            CreateEditProfileFrame,
            BotTrainingFrame,
            MazeBuilderFrame,
            VisualizationFrame,
        ):
            pageName = F.__name__
            frame = F(parent=self.root, controller=self)
            self.frames[pageName] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Default screen
        self.showFrame("ProfileManagementFrame")

    # ----- Navigation helpers -----
    def showFrame(self, pageName: str) -> None:
        frame = self.frames[pageName]
        frame.tkraise()
        if hasattr(frame, 'on_show'):
            try:
                frame.onShow()
            except Exception:
                pass

    def showProfileManagement(self) -> None:
        self.showFrame("ProfileManagementFrame")

    def showCreateEditProfile(self, profile: Any = None) -> None:
        frame = self.frames["CreateEditProfileFrame"]
        frame.loadProfile(profile)
        self.showFrame("CreateEditProfileFrame")

    def showBotTraining(self) -> None:
        self.showFrame("BotTrainingFrame")

    def showVisualizations(self) -> None:
        self.showFrame("VisualizationFrame")

    def showMazeBuilder(self) -> None:
        self.showFrame("MazeBuilderFrame")

    def onClose(self) -> None:
        # Attempt to cancel any legacy polling before destroying root
        try:
            bt = self.frames.get("BotTrainingFrame")
            if bt and hasattr(bt, 'cancel_training_poll'):
                bt.cancelTrainingPoll()
        except Exception:
            pass
        # Request training controller to stop background thread
        try:
            self.trainingController.stop()
        except Exception:
            pass
        # Proactively destroy Matplotlib Tk widgets to avoid Tk finalizer errors
        try:
            viz = self.frames.get("VisualizationFrame")
            if viz is not None and getattr(viz, 'canvas_agg', None):
                try:
                    widget = viz.canvasAgg.get_tk_widget()
                    viz.canvasAgg = None
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
