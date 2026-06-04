from __future__ import annotations


class AppState:
    """Shared UI state accessible to all frames via controller.appState.

    Holds cross-frame state that is too ephemeral for persistence but
    too wide-scoped to live in a single frame widget.
    """

    def __init__(self) -> None:
        self.selected_profile: str = ""
