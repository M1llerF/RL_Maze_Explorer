from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from services.diagnostics import DiagnosticsService

# App-wide event names
PROFILE_SAVED = "profile_saved"
PROFILE_DELETED = "profile_deleted"
TRAINING_RESET = "training_reset"
ARTIFACTS_CLEARED = "artifacts_cleared"
SELECTED_PROFILE_CHANGED = "selected_profile_changed"
FIXED_MAZE_SELECTED = "fixed_maze_selected"


class EventBus:
    """Synchronous publish-subscribe bus for app-wide UI events.

    Emitters call emit(event, **kwargs). Subscribers register with subscribe().
    All handlers run synchronously on the UI thread (Tkinter is single-threaded).
    A failing handler is logged and skipped so it cannot break other subscribers.
    """

    def __init__(self, diagnostics: DiagnosticsService | None = None) -> None:
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        self._diagnostics = diagnostics

    def subscribe(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a callback for an event. Accepts **kwargs from the emitter."""
        self._listeners[event].append(callback)

    def emit(self, event: str, **kwargs: Any) -> None:
        """Notify all subscribers for this event."""
        for cb in list(self._listeners[event]):
            try:
                cb(**kwargs)
            except Exception as e:
                if self._diagnostics is not None:
                    self._diagnostics.exception(
                        "event_bus",
                        "Event handler failed",
                        e,
                        event=event,
                    )
