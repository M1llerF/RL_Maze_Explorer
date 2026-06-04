from __future__ import annotations

import threading
from typing import Any

from botFactory import BotFactory


class BotRuntimeManager:
    """Owns the live bot list and training-session runtime state.

    Responsibilities:
    - Maintain the list of active bot instances.
    - Track per-profile pause/resume state (for visualization overlap).
    - Track per-profile completed episode counts (for UI progress reporting).

    GameEnvironment delegates all of these concerns here so it does not
    accumulate threading primitives and per-profile counters itself.
    """

    def __init__(self, botFactory: BotFactory) -> None:
        self.botFactory = botFactory
        self.bots: list[Any] = []

        self._pauseLock = threading.Lock()
        self._pausedProfiles: set[str] = set()

        self._completedLock = threading.Lock()
        self._completedEpisodes: dict[str, int] = {}

    # ── Bot registration ──────────────────────────────────────────────────────

    def registerBotTypes(self) -> None:
        """Discover and register all available bot types with the factory."""
        from bots import discoverBotClasses
        for botType, botClass in discoverBotClasses().items():
            self.botFactory.registerBot(botType, botClass)

    # ── Pause / resume ────────────────────────────────────────────────────────

    def pause(self, profileName: str) -> None:
        with self._pauseLock:
            self._pausedProfiles.add(profileName)

    def resume(self, profileName: str) -> None:
        with self._pauseLock:
            self._pausedProfiles.discard(profileName)

    def isPaused(self, profileName: str) -> bool:
        with self._pauseLock:
            return profileName in self._pausedProfiles

    # ── Completed episode tracking ────────────────────────────────────────────

    def resetCompleted(self, profileName: str) -> None:
        with self._completedLock:
            self._completedEpisodes[profileName] = 0

    def incCompleted(self, profileName: str) -> None:
        with self._completedLock:
            self._completedEpisodes[profileName] = self._completedEpisodes.get(profileName, 0) + 1

    def getCompleted(self, profileName: str) -> int:
        with self._completedLock:
            return self._completedEpisodes.get(profileName, 0)
