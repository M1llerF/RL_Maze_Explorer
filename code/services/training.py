from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

from gameEnvironment import GameEnvironment
from services.curriculum import CurriculumDecision
from services.diagnostics import DiagnosticsService
from services.trainingSession import TrainingSession

if TYPE_CHECKING:
    from bots.dqnlearning.warmupStore import WarmupStore


class TrainingController:
    """
    Orchestrates background training runs for a selected profile.

    UI passes callbacks for progress, completion, and errors. This separates
    threading and environment control from Tk frame code (SRP, DIP).
    """

    def __init__(
        self,
        env: GameEnvironment,
        diagnostics: DiagnosticsService | None = None,
    ):
        self._env = env
        self._diagnostics = diagnostics
        self._thread: Optional[threading.Thread] = None
        self._collectThread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._session: Optional[TrainingSession] = None

    # Public API ------------------------------------------------------------
    def isActive(self) -> bool:
        return bool(self._session is not None and self._session.is_training)

    def isCollecting(self) -> bool:
        return bool(self._session is not None and self._session.is_collecting)

    @property
    def activeProfile(self) -> Optional[str]:
        if self._session is None:
            return None
        return self._session.selected_profile

    @property
    def currentSession(self) -> Optional[TrainingSession]:
        return self._session

    def collectWarmup(
        self,
        profileName: str,
        mazeMode: str = "Random",
        poolSize: int = 20,
        randomMinGenerationLength: int | None = None,
        randomMaxGenerationLength: int | None = None,
        randomIncludeEnemies: bool = False,
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
            session = TrainingSession(
                selected_profile=profileName,
                maze_mode=mazeMode,
                pool_size=int(poolSize),
                random_min_generation_length=randomMinGenerationLength,
                random_max_generation_length=randomMaxGenerationLength,
                random_include_enemies=bool(randomIncludeEnemies),
                is_collecting=True,
            )
            try:
                session.clear_stop()
                session.configure_environment(self._env)
                self._session = session
            except Exception:
                self._session = None
                raise

            def _collectRun() -> None:
                try:
                    from bots.dqnlearning.warmupStore import WarmupCollector
                    profile = self._env.profileManager.loadProfile(profileName)
                    botIndex = int(self._env.applyProfile(profile, loadCheckpoint=False))
                    session.bind_bot_index(botIndex)
                    bot: Any = self._env.bots[botIndex]
                    store = WarmupCollector().collect(
                        bot=bot,
                        env=self._env,
                        botIndex=botIndex,
                        nTransitions=nTransitions,
                        nCompletions=nCompletions,
                        onProgress=onProgress,
                        onEpisodeComplete=(
                            lambda completedBot: self._handleSessionEpisodeComplete(session, completedBot)
                            if session.curriculum_enabled
                            else None
                        ),
                        stopRequested=session.stop_requested,
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
                        if self._session is session:
                            self._session = None

            self._collectThread = threading.Thread(target=_collectRun, daemon=True)
            self._collectThread.start()

    def stop(self) -> None:
        """Cooperatively stop the current training run."""
        with self._lock:
            session = self._session
            if session is None or not session.is_training:
                return
            session.request_stop()
            # Ask bot to stop mid-episode if it supports it
            prof = session.selected_profile
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
        randomMinGenerationLength: int | None = None,
        randomMaxGenerationLength: int | None = None,
        randomIncludeEnemies: bool = False,
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
            session = TrainingSession(
                selected_profile=profileName,
                maze_mode=mazeMode,
                pool_size=int(poolSize),
                random_min_generation_length=randomMinGenerationLength,
                random_max_generation_length=randomMaxGenerationLength,
                random_include_enemies=bool(randomIncludeEnemies),
                total_rounds=int(rounds),
                is_training=True,
            )
            try:
                session.clear_stop()
                self._session = session
                # Configure maze source
                session.configure_environment(self._env)

                # Load/apply profile & prepare bot
                profile = self._env.profileManager.loadProfile(profileName)
                session.bind_bot_index(int(self._env.applyProfile(profile)))
                # Clear previous completed count
                self._env.resetCompleted(profile.name)
                # Clear stop flags on the bot if present
                if session.bot_index is None:
                    raise RuntimeError("Bot index not set")
                bot = self._env.bots[session.bot_index]
                if hasattr(bot, 'clearStop'):
                    bot.clearStop()
            except Exception:
                self._session = None
                raise

            def _run():
                completedSuccessfully = False
                try:
                    idx0 = session.bot_index
                    if idx0 is None:
                        raise RuntimeError("Bot index not set")
                    trainingBot: Any = self._env.bots[idx0]
                    if warmupStore is not None:
                        n = warmupStore.inject(trainingBot, maxTransitions=warmupTransitionCount)
                        if self._diagnostics is not None:
                            self._diagnostics.info(
                                "training_controller",
                                "Injected warmup transitions",
                                profile_name=profileName,
                                transition_count=int(n),
                            )

                    for i in range(int(rounds)):
                        if session.stop_requested():
                            break
                        # Run a single episode for the selected bot
                        idx = session.bot_index
                        if idx is None:
                            raise RuntimeError("Bot index not set")
                        bot: Any = self._env.bots[idx]

                        # If visualization is open for this profile, pause until resumed
                        while self._env.isTrainingPaused(bot.profileName):
                            if session.stop_requested():
                                break
                            # Simple cooperative yield
                            threading.Event().wait(0.05)
                        if session.stop_requested():
                            break

                        bot.runEpisode()
                        if session.curriculum_enabled:
                            self._handleSessionEpisodeComplete(session, bot)
                        shouldEvaluateFn = getattr(bot, "shouldRunEvaluationEpisode", None)
                        if callable(shouldEvaluateFn):
                            shouldEvaluate = bool(shouldEvaluateFn(i + 1))
                        else:
                            evalFreq = int(getattr(getattr(bot, "config", None), "evaluationFrequency", 0))
                            shouldEvaluate = bool(evalFreq > 0 and (i + 1) % evalFreq == 0)
                        shouldEvaluate = (
                            shouldEvaluate
                            and not getattr(bot, "isWarmingUp", False)
                            and hasattr(bot, "runEvaluationEpisode")
                        )
                        if shouldEvaluate:
                            bot.runEvaluationEpisode()
                        idx2 = session.bot_index
                        if idx2 is None:
                            raise RuntimeError("Bot index not set")
                        self._env.resetEnvironment(idx2)
                        self._env.incCompleted(bot.profileName)
                        session.mark_episode_completed()
                        if onProgress:
                            onProgress(i + 1, rounds)
                    completedSuccessfully = True
                except Exception as e:
                    if onError:
                        onError(e)
                finally:
                    with self._lock:
                        self._thread = None
                        if self._session is session:
                            self._session = None
                    if completedSuccessfully and onComplete:
                        onComplete()

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()

    def _handleSessionEpisodeComplete(
        self,
        session: TrainingSession,
        bot: Any,
    ) -> CurriculumDecision | None:
        decision = session.apply_curriculum_update(self._env, bot)
        if decision is not None and decision.graduated:
            profileName = session.selected_profile
            if self._diagnostics is not None:
                self._diagnostics.info(
                    "training_controller",
                    "Curriculum advanced",
                    profile_name=profileName,
                    previous_size=int(decision.previousSize),
                    next_size=int(decision.nextSize),
                    success_rate=float(decision.successRate),
                    avg_astar_ratio=float(decision.avgAStarRatio),
                )
        return decision
