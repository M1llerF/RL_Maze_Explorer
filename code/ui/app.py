import os
import tkinter as tk
from typing import Any

from services.diagnostics import DiagnosticsService
from gameEnvironment import GameEnvironment
from services.training import TrainingController
from ui.app_state import AppState
from ui.event_bus import EventBus


class MazeAIApp:
    """Main application wiring for Maze AI.

    Responsibilities:
    - Own shared services (GameEnvironment, TrainingController)
    - Build navigation and host frame instances
    - Provide navigation helpers used by frames
    """

    def __init__(
        self,
        root: tk.Tk,
        *,
        game_environment: GameEnvironment | None = None,
        training_controller: TrainingController | None = None,
        event_bus: EventBus | None = None,
        app_state: AppState | None = None,
        diagnostics: DiagnosticsService | None = None,
    ):
        self.root = root
        self.diagnostics = diagnostics
        self.root.title("Maze AI Experiment")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        # Ensure a default folder exists for user mazes
        try:
            os.makedirs('mazes', exist_ok=True)
            os.makedirs('profiles', exist_ok=True)
        except Exception:
            pass

        self.game_environment = game_environment or GameEnvironment(diagnostics=self.diagnostics)
        self.gameEnv = self.game_environment
        self.training_controller = training_controller or TrainingController(
            self.game_environment,
            diagnostics=self.diagnostics,
        )
        self.trainingController = self.training_controller
        self.event_bus = event_bus or EventBus(diagnostics=self.diagnostics)
        self.eventBus = self.event_bus
        self.app_state = app_state or AppState()
        self.appState = self.app_state

        self.create_navigation_bar()
        self.create_main_frames()
        # Graceful shutdown handler to cancel pending timers
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----- UI building -----
    def create_navigation_bar(self) -> None:
        self.menuBar = tk.Menu(self.root)
        self.root.config(menu=self.menuBar)

        self.navMenu = tk.Menu(self.menuBar, tearoff=0)
        self.menuBar.add_cascade(label="Navigation", menu=self.navMenu)
        self.navMenu.add_command(label="Profile Management", command=self.show_profile_management)
        self.navMenu.add_command(label="Bot Training", command=self.show_bot_training)
        self.navMenu.add_command(label="Maze Builder", command=self.show_maze_builder)
        self.navMenu.add_command(label="Visualizations", command=self.show_visualizations)
        self.navMenu.add_command(label="Exit", command=self.root.quit)

    def createNavigationBar(self) -> None:
        self.create_navigation_bar()

    def create_main_frames(self) -> None:
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
        self.show_frame("ProfileManagementFrame")

    def createMainFrames(self) -> None:
        self.create_main_frames()

    # ----- Navigation helpers -----
    def show_frame(self, pageName: str) -> None:
        frame = self.frames[pageName]
        frame.tkraise()
        if hasattr(frame, 'on_show'):
            frame.on_show()

    def showFrame(self, pageName: str) -> None:
        self.show_frame(pageName)

    def show_profile_management(self) -> None:
        self.show_frame("ProfileManagementFrame")

    def showProfileManagement(self) -> None:
        self.show_profile_management()

    def show_create_edit_profile(self, profile: Any = None) -> None:
        frame = self.frames["CreateEditProfileFrame"]
        load_profile = getattr(frame, "load_profile", None) or getattr(frame, "loadProfile")
        load_profile(profile)
        self.show_frame("CreateEditProfileFrame")

    def showCreateEditProfile(self, profile: Any = None) -> None:
        self.show_create_edit_profile(profile)

    def show_bot_training(self) -> None:
        self.show_frame("BotTrainingFrame")

    def showBotTraining(self) -> None:
        self.show_bot_training()

    def show_visualizations(self) -> None:
        self.show_frame("VisualizationFrame")

    def showVisualizations(self) -> None:
        self.show_visualizations()

    def show_maze_builder(self) -> None:
        self.show_frame("MazeBuilderFrame")

    def showMazeBuilder(self) -> None:
        self.show_maze_builder()

    def on_close(self) -> None:
        # Attempt to cancel any legacy polling before destroying root
        try:
            bt = self.frames.get("BotTrainingFrame")
            if bt:
                cancel_poll = getattr(bt, "cancel_training_poll", None) or getattr(bt, "cancelTrainingPoll", None)
                if callable(cancel_poll):
                    cancel_poll()
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
            canvas_agg = None if viz is None else (
                getattr(viz, "canvas_agg", None) or getattr(viz, "canvasAgg", None)
            )
            if viz is not None and canvas_agg is not None:
                try:
                    widget = canvas_agg.get_tk_widget()
                    if hasattr(viz, "canvas_agg"):
                        viz.canvas_agg = None
                    if hasattr(viz, "canvasAgg"):
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

    def onClose(self) -> None:
        self.on_close()
