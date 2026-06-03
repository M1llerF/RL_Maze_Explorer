from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from services.repository import ArtifactsRepository
from .replay import UniformReplayStore
from .types import Transition


@dataclass(frozen=True)
class WarmupStoreMeta:
    encoderFingerprint: str
    inputDim: int
    transitionCount: int


@dataclass(frozen=True)
class WarmupStoreOption:
    profileName: str
    transitionCount: int
    store: "WarmupStore"
    isLocal: bool = False

    @property
    def label(self) -> str:
        localSuffix = " [local]" if self.isLocal else ""
        return f"{self.profileName} ({self.transitionCount} transitions){localSuffix}"


class WarmupStore:
    """
    Pre-collected planner demonstration transitions, saveable and reusable
    across training sessions.

    Transitions are stored as stacked numpy arrays (not per-entry Python dicts)
    so save/load is fast even for large stores.

    A store is tied to an encoder fingerprint and input dimensionality — it can
    only be injected into a bot whose encoder produces the same fingerprint.
    Attempting to inject an incompatible store raises ValueError.
    """

    ARTIFACT_NAME = "warmup.pt"
    _PAYLOAD_VERSION = 1

    def __init__(
        self,
        meta: WarmupStoreMeta,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        nextStates: np.ndarray,
        dones: np.ndarray,
        masks: np.ndarray,
        nextMasks: np.ndarray,
    ) -> None:
        self.meta = meta
        self._states = states           # (N, D) float32
        self._actions = actions         # (N,) int64
        self._rewards = rewards         # (N,) float32
        self._nextStates = nextStates   # (N, D) float32
        self._dones = dones             # (N,) float32
        self._masks = masks             # (N, 4) bool
        self._nextMasks = nextMasks     # (N, 4) bool

    @property
    def transitionCount(self) -> int:
        return int(self._states.shape[0])

    def isCompatibleWith(self, fingerprint: str, inputDim: int) -> bool:
        return (
            self.meta.encoderFingerprint == fingerprint
            and self.meta.inputDim == inputDim
        )

    def inject(self, bot: Any, maxTransitions: int | None = None) -> int:
        """
        Pre-fill bot.agent.replay with warmup transitions.

        After injection len(replay) >= replayWarmupSteps, so the bot's
        chooseAction skips the built-in planner warmup and DQN training
        begins immediately on the first episode.

        Returns the number of transitions actually pushed (capped by replay
        capacity).
        """
        if not self.isCompatibleWith(
            bot.checkpointMeta.encoderConfigFingerprint,
            bot.checkpointMeta.inputDim,
        ):
            raise ValueError(
                "WarmupStore is incompatible with this bot's encoder "
                f"(fingerprint or inputDim mismatch)."
            )
        requested = self.transitionCount if maxTransitions is None else max(0, int(maxTransitions))
        N = min(self.transitionCount, bot.agent.replay.capacity, requested)
        for i in range(N):
            bot.agent.replay.push(
                Transition(
                    state=self._states[i],
                    action=int(self._actions[i]),
                    reward=float(self._rewards[i]),
                    nextState=self._nextStates[i],
                    done=bool(self._dones[i]),
                    validActionMask=self._masks[i],
                    nextValidActionMask=self._nextMasks[i],
                )
            )
        return N

    def save(self, repo: ArtifactsRepository, profileName: str) -> None:
        payload: dict[str, Any] = {
            "version": self._PAYLOAD_VERSION,
            "fingerprint": self.meta.encoderFingerprint,
            "input_dim": self.meta.inputDim,
            "transition_count": self.transitionCount,
            "states": self._states,
            "actions": self._actions,
            "rewards": self._rewards,
            "next_states": self._nextStates,
            "dones": self._dones,
            "masks": self._masks,
            "next_masks": self._nextMasks,
        }
        buf = io.BytesIO()
        torch.save(payload, buf)
        repo.saveModelArtifactBytes(profileName, self.ARTIFACT_NAME, buf.getvalue())

    @classmethod
    def load(cls, repo: ArtifactsRepository, profileName: str) -> WarmupStore | None:
        """Return None when no store exists or the file is unreadable."""
        raw = repo.loadModelArtifactBytes(profileName, cls.ARTIFACT_NAME)
        if raw is None:
            return None
        try:
            data = torch.load(
                io.BytesIO(raw), map_location="cpu", weights_only=False
            )
            if int(data.get("version", 0)) != cls._PAYLOAD_VERSION:
                return None
            meta = WarmupStoreMeta(
                encoderFingerprint=str(data["fingerprint"]),
                inputDim=int(data["input_dim"]),
                transitionCount=int(data["transition_count"]),
            )
            return cls(
                meta=meta,
                states=np.asarray(data["states"], dtype=np.float32),
                actions=np.asarray(data["actions"], dtype=np.int64),
                rewards=np.asarray(data["rewards"], dtype=np.float32),
                nextStates=np.asarray(data["next_states"], dtype=np.float32),
                dones=np.asarray(data["dones"], dtype=np.float32),
                masks=np.asarray(data["masks"], dtype=np.bool_),
                nextMasks=np.asarray(data["next_masks"], dtype=np.bool_),
            )
        except Exception:
            return None

    @classmethod
    def exists(cls, repo: ArtifactsRepository, profileName: str) -> bool:
        return repo.loadModelArtifactBytes(profileName, cls.ARTIFACT_NAME) is not None

    def clear(self, repo: ArtifactsRepository, profileName: str) -> None:
        """Delete the warmup artifact for this profile."""
        import os
        path = os.path.join(repo.modelArtifactsDir(profileName), self.ARTIFACT_NAME)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    @classmethod
    def inferCompatibilityForProfile(
        cls,
        profile: Any,
    ) -> tuple[str, int] | None:
        if getattr(profile, "botType", "") != "DQNBot":
            return None
        config = getattr(profile, "config", None)
        if config is None:
            return None

        from .encoder import StateEncoder
        from .model import MAP_CHANNELS

        useRichEncoding = bool(getattr(config, "useRichEncoding", False))
        mapHeight = int(getattr(config, "neuralMapHeight", 31))
        mapWidth = int(getattr(config, "neuralMapWidth", 31))
        neuralMap: tuple[float, ...] = ()
        if useRichEncoding:
            neuralMap = tuple(0.0 for _ in range(MAP_CHANNELS * mapHeight * mapWidth))

        sampleObservation = (
            (0, 0),
            (0, 0, 0, 0),
            (0, 0),
            -1,
            ((0.0, 0.0, 0.0),) * 4,
            neuralMap,
            (1, 1, 1, 1),
            (0.0, 0.0, 0.0),
        )
        schema = StateEncoder(
            config,
            mazeHeight=1,
            mazeWidth=1,
        ).inferSchema(sampleObservation)
        return schema.encoderConfigFingerprint, schema.inputDim

    @classmethod
    def loadCompatibleOptions(
        cls,
        repo: ArtifactsRepository,
        profileNames: list[str],
        *,
        selectedProfile: str,
        fingerprint: str,
        inputDim: int,
    ) -> list[WarmupStoreOption]:
        options: list[WarmupStoreOption] = []
        for profileName in profileNames:
            store = cls.load(repo, profileName)
            if store is None or not store.isCompatibleWith(fingerprint, inputDim):
                continue
            options.append(
                WarmupStoreOption(
                    profileName=profileName,
                    transitionCount=store.transitionCount,
                    store=store,
                    isLocal=(profileName == selectedProfile),
                )
            )
        options.sort(
            key=lambda option: (
                0 if option.isLocal else 1,
                -option.transitionCount,
                option.profileName.lower(),
            )
        )
        return options


class WarmupCollector:
    """
    Runs a DQN bot in planner-only mode to build a WarmupStore.

    Pass either nTransitions (stop after N steps collected) or nCompletions
    (stop after N successful maze solves).  Exactly one must be supplied;
    nTransitions is the default when neither is given.
    """

    def collect(
        self,
        bot: Any,
        env: Any,
        botIndex: int,
        nTransitions: int | None = None,
        nCompletions: int | None = None,
        onProgress: Callable[[int, int], None] | None = None,
        onEpisodeComplete: Callable[[Any], None] | None = None,
        stopRequested: Callable[[], bool] | None = None,
    ) -> WarmupStore:
        """
        Collect planner-guided transitions and return a WarmupStore.

        nTransitions — stop once this many transitions have been collected
            (defaults to config.replayWarmupSteps when neither arg is given).
        nCompletions — stop once this many mazes have been successfully solved.
            The replay is sized to replayCapacity so no transitions are dropped.

        onProgress receives (current, total) where both quantities match the
        chosen mode (transitions or completions).
        """
        if nCompletions is not None:
            return self._collectByCompletions(
                bot, env, botIndex, nCompletions, onProgress, onEpisodeComplete, stopRequested
            )
        return self._collectByTransitions(
            bot, env, botIndex, nTransitions, onProgress, onEpisodeComplete, stopRequested
        )

    def _collectByTransitions(
        self,
        bot: Any,
        env: Any,
        botIndex: int,
        nTransitions: int | None,
        onProgress: Callable[[int, int], None] | None,
        onEpisodeComplete: Callable[[Any], None] | None,
        stopRequested: Callable[[], bool] | None,
    ) -> WarmupStore:
        maxTransitions = int(getattr(bot.config, "replayWarmupSteps", 2000))
        target = maxTransitions if nTransitions is None else max(1, int(nTransitions))

        originalReplay = bot.agent.replay
        collectionReplay = UniformReplayStore(capacity=target)
        bot.agent.replay = collectionReplay
        previousWarmupMode = bool(getattr(bot, "_collectingWarmup", False))
        bot._collectingWarmup = True

        try:
            while len(bot.agent.replay) < target:
                if stopRequested is not None and stopRequested():
                    break
                bot.runEpisode()
                if onEpisodeComplete is not None:
                    onEpisodeComplete(bot)
                if onProgress is not None:
                    onProgress(len(bot.agent.replay), target)
                env.resetEnvironment(botIndex)
                time.sleep(0.01)
        finally:
            bot._collectingWarmup = previousWarmupMode
            bot.agent.replay = originalReplay

        return self._buildStore(bot, collectionReplay)

    def _collectByCompletions(
        self,
        bot: Any,
        env: Any,
        botIndex: int,
        nCompletions: int,
        onProgress: Callable[[int, int], None] | None,
        onEpisodeComplete: Callable[[Any], None] | None,
        stopRequested: Callable[[], bool] | None,
    ) -> WarmupStore:
        target = max(1, int(nCompletions))
        replayCapacity = int(getattr(bot.config, "replayCapacity", 50000))

        originalReplay = bot.agent.replay
        collectionReplay = UniformReplayStore(capacity=replayCapacity)
        bot.agent.replay = collectionReplay
        previousWarmupMode = bool(getattr(bot, "_collectingWarmup", False))
        bot._collectingWarmup = True

        completed = 0
        try:
            while completed < target:
                if stopRequested is not None and stopRequested():
                    break
                bot.runEpisode()
                if getattr(bot, "lastEpisodeSuccess", False):
                    completed += 1
                if onEpisodeComplete is not None:
                    onEpisodeComplete(bot)
                if onProgress is not None:
                    onProgress(completed, target)
                env.resetEnvironment(botIndex)
                time.sleep(0.01)
        finally:
            bot._collectingWarmup = previousWarmupMode
            bot.agent.replay = originalReplay

        return self._buildStore(bot, collectionReplay)

    @staticmethod
    def _buildStore(bot: Any, collectionReplay: Any) -> "WarmupStore":
        transitions = [
            collectionReplay.buffer[i]
            for i in range(collectionReplay.size)
            if collectionReplay.buffer[i] is not None
        ]
        if not transitions:
            raise RuntimeError("No transitions were collected.")

        meta = WarmupStoreMeta(
            encoderFingerprint=bot.checkpointMeta.encoderConfigFingerprint,
            inputDim=bot.checkpointMeta.inputDim,
            transitionCount=len(transitions),
        )
        return WarmupStore(
            meta=meta,
            states=np.stack([t.state for t in transitions]).astype(np.float32),
            actions=np.array([t.action for t in transitions], dtype=np.int64),
            rewards=np.array([t.reward for t in transitions], dtype=np.float32),
            nextStates=np.stack([t.nextState for t in transitions]).astype(np.float32),
            dones=np.array([float(t.done) for t in transitions], dtype=np.float32),
            masks=np.stack([t.validActionMask for t in transitions]).astype(np.bool_),
            nextMasks=np.stack([t.nextValidActionMask for t in transitions]).astype(np.bool_),
        )
