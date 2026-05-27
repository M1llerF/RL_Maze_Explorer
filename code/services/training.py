from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from gameEnvironment import GameEnvironment


class TrainingController:
    """
    Orchestrates background training runs for a selected profile.

    UI passes callbacks for progress, completion, and errors. This separates
    threading and environment control from Tk frame code (SRP, DIP).
    """

    def __init__(self, env: GameEnvironment):
        self._env = env
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._activeProfile: Optional[str] = None
        self._botIndex: Optional[int] = None

    # Public API ------------------------------------------------------------
    def isActive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def activeProfile(self) -> Optional[str]:
        return self._activeProfile

    def stop(self) -> None:
        """Cooperatively stop the current training run."""
        with self._lock:
            if not self.isActive():
                return
            self._stop.set()
            # Ask bot to stop mid-episode if it supports it
            try:
                prof = self._activeProfile
                if prof:
                    for b in self._env.bots:
                        if getattr(b, 'profile_name', None) == prof and hasattr(b, 'request_stop'):
                            b.requestStop()
                            break
            except Exception:
                pass

    def start(
        self,
        profileName: str,
        rounds: int,
        mazeMode: str = "Random",
        poolSize: int = 20,
        onProgress: Optional[Callable[[int, int], None]] = None,
        onError: Optional[Callable[[Exception], None]] = None,
        onComplete: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Start a background training run for the given profile.

        - profile_name: name of the profile to load/apply
        - rounds: number of episodes to run
        - maze_mode: one of "Random", "Fixed (Builder)", or "Pool"
        - pool_size: number of fixed mazes when in Pool mode
        - callbacks: invoked on main thread by caller (UI should .after(...) if needed)
        """
        with self._lock:
            if self.isActive():
                return
            self._stop.clear()
            self._activeProfile = profileName
            self._botIndex = None
            # Configure maze source
            try:
                if mazeMode == "Pool":
                    self._env.configureTrainingPool(size=int(poolSize))
                elif mazeMode == "Fixed (Builder)":
                    # Ensure a fixed maze is active; let UI pre-set it
                    if not self._env.fixedMazeActive or self._env.fixedMazeState is None:
                        raise RuntimeError("Fixed maze is not configured.")
                    # Training pool must be off when fixed maze is used
                    self._env.trainingPoolActive = False
                else:
                    # Random each reset
                    self._env.trainingPoolActive = False
                    self._env.fixedMazeActive = False
                    self._env.fixedMazeState = None
            except Exception as e:
                if onError:
                    onError(e)
                return

            # Load/apply profile & prepare bot
            try:
                profile = self._env.profileManager.loadProfile(profileName)
                self._botIndex = int(self._env.applyProfile(profile))
                # Clear previous completed count
                self._env.resetCompleted(profile.name)
                # Clear stop flags on the bot if present
                try:
                    bot = self._env.bots[self._botIndex]
                    if hasattr(bot, 'clear_stop'):
                        bot.clearStop()
                except Exception:
                    pass
            except Exception as e:
                if onError:
                    onError(e)
                return

            def _run():
                try:
                    for i in range(int(rounds)):
                        if self._stop.is_set():
                            break
                        # Run a single episode for the selected bot
                        try:
                            idx = self._botIndex
                            if idx is None:
                                raise RuntimeError("Bot index not set")
                            bot: Any = self._env.bots[idx]
                        except Exception as exc:
                            raise RuntimeError("Bot not available") from exc

                        # If visualization is open for this profile, pause until resumed
                        while self._env.isTrainingPaused(bot.profileName):
                            if self._stop.is_set():
                                break
                            # Simple cooperative yield
                            threading.Event().wait(0.05)
                        if self._stop.is_set():
                            break

                        bot.runEpisode()
                        idx2 = self._botIndex
                        if idx2 is None:
                            raise RuntimeError("Bot index not set")
                        self._env.resetEnvironment(idx2)
                        self._env.incCompleted(bot.profileName)
                        if onProgress:
                            try:
                                onProgress(i + 1, rounds)
                            except Exception:
                                pass
                except Exception as e:
                    if onError:
                        onError(e)
                finally:
                    with self._lock:
                        self._thread = None
                        self._stop.clear()
                        self._activeProfile = None
                        self._botIndex = None
                    if onComplete:
                        try:
                            onComplete()
                        except Exception:
                            pass

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()

