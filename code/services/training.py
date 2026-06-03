from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

from gameEnvironment import GameEnvironment
from services.curriculum import CurriculumConfig, CurriculumManager

if TYPE_CHECKING:
    from bots.dqnlearning.warmup_store import WarmupStore


class TrainingController:
    """
    Orchestrates background training runs for a selected profile.

    UI passes callbacks for progress, completion, and errors. This separates
    threading and environment control from Tk frame code (SRP, DIP).
    """

    def __init__(self, env: GameEnvironment):
        self._env = env
        self._thread: Optional[threading.Thread] = None
        self._collectThread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._activeProfile: Optional[str] = None
        self._botIndex: Optional[int] = None
        self._curriculum = CurriculumManager(
            CurriculumConfig(
                windowSize=20,
                minSuccessRateToGraduate=0.7,
                maxAStarRatioToGraduate=2.0,
                maxMazeSize=31,
            )
        )

    # Public API ------------------------------------------------------------
    def isActive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def isCollecting(self) -> bool:
        return self._collectThread is not None and self._collectThread.is_alive()

    @property
    def activeProfile(self) -> Optional[str]:
        return self._activeProfile

    def collectWarmup(
        self,
        profileName: str,
        mazeMode: str = "Random",
        poolSize: int = 20,
        nTransitions: int | None = None,
        nCompletions: int | None = None,
        onProgress: Optional[Callable[[int, int], None]] = None,
        onComplete: Optional[Callable[["WarmupStore"], None]] = None,
        onError: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Collect planner-guided warmup transitions in a background thread,
        then save the resulting WarmupStore to the profile's artifact directory.
        """
        with self._lock:
            if self.isActive() or self.isCollecting():
                return
            self._stop.clear()
            self._activeProfile = profileName

            if mazeMode == "Pool":
                self._env.configureTrainingPool(size=int(poolSize))
                self._env.curriculumActive = False
            elif mazeMode == "Fixed (Builder)":
                if not self._env.fixedMazeActive or self._env.fixedMazeState is None:
                    raise RuntimeError("Fixed maze is not configured.")
                self._env.trainingPoolActive = False
                self._env.curriculumActive = False
            else:
                self._env.trainingPoolActive = False
                self._env.fixedMazeActive = False
                self._env.fixedMazeState = None
                self._env.curriculumActive = True

            def _collectRun() -> None:
                try:
                    from bots.dqnlearning.warmup_store import WarmupCollector
                    profile = self._env.profileManager.loadProfile(profileName)
                    botIndex = int(self._env.applyProfile(profile, loadCheckpoint=False))
                    self._botIndex = botIndex
                    bot: Any = self._env.bots[botIndex]
                    if self._env.curriculumActive:
                        width, height = self._env.getMazeSize()
                        startSize = min(int(width), int(height))
                        self._curriculum.resetProfile(profile.name, initialSize=startSize)
                    store = WarmupCollector().collect(
                        bot=bot,
                        env=self._env,
                        botIndex=botIndex,
                        nTransitions=nTransitions,
                        nCompletions=nCompletions,
                        onProgress=onProgress,
                        onEpisodeComplete=(
                            self._updateCurriculumForEpisode
                            if self._env.curriculumActive
                            else None
                        ),
                        stopRequested=self._stop.is_set,
                    )
                    store.save(self._env.repository, profileName)
                    if onComplete:
                        onComplete(store)
                except Exception as exc:
                    if onError:
                        onError(exc)
                finally:
                    with self._lock:
                        self._collectThread = None
                        self._stop.clear()
                        self._activeProfile = None
                        self._botIndex = None

            self._collectThread = threading.Thread(target=_collectRun, daemon=True)
            self._collectThread.start()

    def stop(self) -> None:
        """Cooperatively stop the current training run."""
        with self._lock:
            if not self.isActive():
                return
            self._stop.set()
            # Ask bot to stop mid-episode if it supports it
            prof = self._activeProfile
            if prof:
                for b in self._env.bots:
                    if getattr(b, 'profileName', None) == prof and hasattr(b, 'requestStop'):
                        b.requestStop()
                        break

    def start(
        self,
        profileName: str,
        rounds: int,
        mazeMode: str = "Random",
        poolSize: int = 20,
        warmupStore: Optional["WarmupStore"] = None,
        warmupTransitionCount: int | None = None,
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
            if mazeMode == "Pool":
                self._env.configureTrainingPool(size=int(poolSize))
            elif mazeMode == "Fixed (Builder)":
                # Ensure a fixed maze is active; let UI pre-set it
                if not self._env.fixedMazeActive or self._env.fixedMazeState is None:
                    raise RuntimeError("Fixed maze is not configured.")
                # Training pool must be off when fixed maze is used
                self._env.trainingPoolActive = False
                self._env.curriculumActive = False
            else:
                # Random each reset
                self._env.trainingPoolActive = False
                self._env.fixedMazeActive = False
                self._env.fixedMazeState = None
                self._env.curriculumActive = True

            # Load/apply profile & prepare bot
            profile = self._env.profileManager.loadProfile(profileName)
            self._botIndex = int(self._env.applyProfile(profile))
            # Clear previous completed count
            self._env.resetCompleted(profile.name)
            # Clear stop flags on the bot if present
            bot = self._env.bots[self._botIndex]
            if hasattr(bot, 'clearStop'):
                bot.clearStop()
            if self._env.curriculumActive:
                width, height = self._env.getMazeSize()
                startSize = min(int(width), int(height))
                self._curriculum.resetProfile(profile.name, initialSize=startSize)

            def _run():
                completedSuccessfully = False
                try:
                    idx0 = self._botIndex
                    if idx0 is None:
                        raise RuntimeError("Bot index not set")
                    trainingBot: Any = self._env.bots[idx0]
                    if warmupStore is not None:
                        try:
                            n = warmupStore.inject(trainingBot, maxTransitions=warmupTransitionCount)
                            print(f"[Warmup] Injected {n} transitions — skipping built-in warmup.")
                        except Exception as exc:
                            print(f"[Warmup] Injection skipped: {exc}")

                    for i in range(int(rounds)):
                        if self._stop.is_set():
                            break
                        # Run a single episode for the selected bot
                        idx = self._botIndex
                        if idx is None:
                            raise RuntimeError("Bot index not set")
                        bot: Any = self._env.bots[idx]

                        # If visualization is open for this profile, pause until resumed
                        while self._env.isTrainingPaused(bot.profileName):
                            if self._stop.is_set():
                                break
                            # Simple cooperative yield
                            threading.Event().wait(0.05)
                        if self._stop.is_set():
                            break

                        bot.runEpisode()
                        if self._env.curriculumActive:
                            self._updateCurriculumForEpisode(bot)
                        idx2 = self._botIndex
                        if idx2 is None:
                            raise RuntimeError("Bot index not set")
                        self._env.resetEnvironment(idx2)
                        self._env.incCompleted(bot.profileName)
                        if onProgress:
                            onProgress(i + 1, rounds)
                    completedSuccessfully = True
                except Exception as e:
                    if onError:
                        onError(e)
                finally:
                    with self._lock:
                        self._thread = None
                        self._stop.clear()
                        self._activeProfile = None
                        self._botIndex = None
                    if completedSuccessfully and onComplete:
                        onComplete()

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()

    def _updateCurriculumForEpisode(self, bot: Any) -> None:
        success = bool(getattr(bot, "lastEpisodeSuccess", False))
        episodeSteps = int(getattr(bot, "lastEpisodeSteps", 0))
        astarSteps = int(getattr(bot, "lastEpisodeOptimalSteps", 0))
        profileName = str(getattr(bot, "profileName", ""))
        if not profileName:
            return
        decision = self._curriculum.update(
            profileName,
            success=success,
            episodeSteps=episodeSteps,
            astarSteps=astarSteps,
        )
        if decision.graduated:
            nextSize = int(decision.nextSize)
            self._env.setMazeSize(nextSize, nextSize)
            print(
                f"[Curriculum] {profileName}: size {decision.previousSize}->{decision.nextSize} "
                f"(successRate={decision.successRate:.2f}, avgAStarRatio={decision.avgAStarRatio:.2f})"
            )
