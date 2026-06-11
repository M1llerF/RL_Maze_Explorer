from __future__ import annotations

import os
import tkinter as tk
from dataclasses import dataclass

from defaultProfiles import ensure_default_profiles
from gameEnvironment import GameEnvironment
from services.diagnostics import DiagnosticsService
from services.training import TrainingController
from ui.app import MazeAIApp
from ui.appState import AppState
from ui.eventBus import EventBus


@dataclass(frozen=True)
class AppServices:
    diagnostics: DiagnosticsService
    game_environment: GameEnvironment
    training_controller: TrainingController
    event_bus: EventBus
    app_state: AppState


def ensure_app_directories() -> None:
    for path in ("mazes", "profiles"):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            continue


def build_app_services() -> AppServices:
    ensure_app_directories()
    diagnostics = DiagnosticsService()
    game_environment = GameEnvironment(diagnostics=diagnostics)
    ensure_default_profiles(game_environment.profileManager, game_environment.repository)
    training_controller = TrainingController(game_environment, diagnostics=diagnostics)
    event_bus = EventBus(diagnostics=diagnostics)
    app_state = AppState()
    return AppServices(
        diagnostics=diagnostics,
        game_environment=game_environment,
        training_controller=training_controller,
        event_bus=event_bus,
        app_state=app_state,
    )


def create_app(root: tk.Tk) -> MazeAIApp:
    services = build_app_services()
    return MazeAIApp(
        root,
        game_environment=services.game_environment,
        training_controller=services.training_controller,
        event_bus=services.event_bus,
        app_state=services.app_state,
        diagnostics=services.diagnostics,
    )
