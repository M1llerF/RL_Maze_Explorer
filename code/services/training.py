from __future__ import annotations

import threading
from typing import Callable, Optional

from BotProfile import BotProfile
from GameEnvironment import GameEnvironment


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
        self._active_profile: Optional[str] = None
        self._bot_index: Optional[int] = None

    # Public API ------------------------------------------------------------
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def active_profile(self) -> Optional[str]:
        return self._active_profile

    def stop(self) -> None:
        """Cooperatively stop the current training run."""
        with self._lock:
            if not self.is_active():
                return
            self._stop.set()
            # Ask bot to stop mid-episode if it supports it
            try:
                prof = self._active_profile
                if prof:
                    for b in self._env.bots:
                        if getattr(b, 'profile_name', None) == prof and hasattr(b, 'request_stop'):
                            b.request_stop()
                            break
            except Exception:
                pass

    def start(
        self,
        profile_name: str,
        rounds: int,
        maze_mode: str = "Random",
        pool_size: int = 20,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
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
            if self.is_active():
                return
            self._stop.clear()
            self._active_profile = profile_name
            self._bot_index = None
            # Configure maze source
            try:
                if maze_mode == "Pool":
                    self._env.configure_training_pool(size=int(pool_size))
                elif maze_mode == "Fixed (Builder)":
                    # Ensure a fixed maze is active; let UI pre-set it
                    if not self._env.fixed_maze_active or self._env.fixed_maze_state is None:
                        raise RuntimeError("Fixed maze is not configured.")
                    # Training pool must be off when fixed maze is used
                    self._env.training_pool_active = False
                else:
                    # Random each reset
                    self._env.training_pool_active = False
                    self._env.fixed_maze_active = False
                    self._env.fixed_maze_state = None
            except Exception as e:
                if on_error:
                    on_error(e)
                return

            # Load/apply profile & prepare bot
            try:
                profile = self._env.profile_manager.load_profile(profile_name)
                self._bot_index = self._env.apply_profile(profile)
                # Clear previous completed count
                self._env.reset_completed(profile.name)
                # Clear stop flags on the bot if present
                try:
                    bot = self._env.bots[self._bot_index]
                    if hasattr(bot, 'clear_stop'):
                        bot.clear_stop()
                except Exception:
                    pass
            except Exception as e:
                if on_error:
                    on_error(e)
                return

            def _run():
                try:
                    for i in range(int(rounds)):
                        if self._stop.is_set():
                            break
                        # Run a single episode for the selected bot
                        try:
                            bot = self._env.bots[self._bot_index]
                        except Exception as exc:
                            raise RuntimeError("Bot not available") from exc

                        # If visualization is open for this profile, pause until resumed
                        while self._env.is_training_paused(bot.profile_name):
                            if self._stop.is_set():
                                break
                            # Simple cooperative yield
                            threading.Event().wait(0.05)
                        if self._stop.is_set():
                            break

                        bot.run_episode()
                        self._env.reset_environment(self._bot_index)
                        self._env.inc_completed(bot.profile_name)
                        if on_progress:
                            try:
                                on_progress(i + 1, rounds)
                            except Exception:
                                pass
                except Exception as e:
                    if on_error:
                        on_error(e)
                finally:
                    with self._lock:
                        self._thread = None
                        self._stop.clear()
                        self._active_profile = None
                        self._bot_index = None
                    if on_complete:
                        try:
                            on_complete()
                        except Exception:
                            pass

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()

