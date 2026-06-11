from __future__ import annotations

from typing import Any, TYPE_CHECKING

from services.episodeResult import EpisodePersistencePolicy, EvaluationEpisodeDefinition

if TYPE_CHECKING:
    from bots.botStatus import BotStatus
    from bots.common.decision import LocalActionSpace


class BaseBot:
    """
    Base lifecycle contract for training and visualization.

    Training lifecycle:
    - `run_episode()` is a full training episode entrypoint.
    - Runners may call `on_episode_start(...)`, `on_episode_step(...)`,
      and `on_episode_end(...)` hooks around episode execution.
    - Training updates and persistence are allowed only in training flows.

    Visualization lifecycle:
    - Visualization must remain inference-only and non-persistent.
    - Bots can expose step-wise visualization methods, but these must not
      mutate durable training artifacts.

    Action / environment contract:
    - Subclasses that use ActionRegistry must implement encodeState(),
      getActionSpace(), getPolicyActionSpace(), getValidActionMask(),
      applyStep(), addReward(), shapingPenalty(), and isWarmingUp.
    - The runner/agent interacts with the bot only through these methods;
      it never knows what any action means internally.
    """
    def __init__(self, maze: Any, statistics: Any, config: Any = None) -> None:
        self.maze = maze
        self.statistics = statistics
        self.config = config
        self._stopRequested = False
        self._pushCooldownRemaining: int = 0

    # ── Push cooldown ─────────────────────────────────────────────────────────

    @property
    def pushCooldownSteps(self) -> int:
        return max(1, int(getattr(self.config, "pushCooldownSteps", 3)))

    def isPushReady(self) -> bool:
        return self._pushCooldownRemaining <= 0

    def consumePushCooldown(self) -> None:
        self._pushCooldownRemaining = self.pushCooldownSteps

    def tickPushCooldown(self) -> None:
        if self._pushCooldownRemaining > 0:
            self._pushCooldownRemaining -= 1

    def autoAttackAdjacentEnemyLocalId(self, actionSpace: "LocalActionSpace") -> int | None:
        if not getattr(self.config, "autoAttackAdjacentEnemy", False):
            return None
        if not getattr(self.config, "useAttackActions", False):
            return None
        context = getattr(self, "_context", None)
        if context is None:
            return None

        from bots.common.actions import ATTACK_UP, ATTACK_DOWN, ATTACK_LEFT, ATTACK_RIGHT, DIRECTION_DELTAS
        from environment.entities import Enemy

        attackDirections = (ATTACK_UP, ATTACK_DOWN, ATTACK_LEFT, ATTACK_RIGHT)
        for semanticId in attackDirections:
            localId = actionSpace.localId(semanticId)
            if localId is None:
                continue
            if not actionSpace.validActionMask[localId]:
                continue
            dr, dc = DIRECTION_DELTAS[int(semanticId)]
            target = (int(self.position[0]) + dr, int(self.position[1]) + dc)
            occupants = getattr(context, "entitiesAt", lambda _target: [])(target)
            for entity in occupants:
                if isinstance(entity, Enemy) and getattr(entity, "_alive", True):
                    return localId
        return None

    def reset(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def calculateState(self) -> Any:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def runEpisode(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    # ── Episode lifecycle hooks ───────────────────────────────────────────────

    def onEpisodeStart(self, mode: str) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def onEpisodeStep(self, mode: str, stepIndex: int) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def onEpisodeEnd(self, mode: str, outcome: str) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    # ── Shared action / environment contract ─────────────────────────────────
    # Subclasses using ActionRegistry should override these.

    def encodeState(self) -> Any:
        """Return the encoded state ready for the learning algorithm."""
        raise NotImplementedError

    def getActionSpace(self) -> list[Any]:
        """Return list[ActionSpec] from the bot's ActionRegistry."""
        raise NotImplementedError

    def getPolicyActionSpace(self) -> list[Any]:
        """
        Return the bot's current policy action space as dense local ActionSpecs.

        This ordering must match any learned value table indexed by local action
        id, so visualization and diagnostics can label actions correctly when
        macro-actions/options are enabled.
        """
        raise NotImplementedError

    def getValidActionMask(self) -> list[bool]:
        """Return a bool mask over the full action space for the current state."""
        raise NotImplementedError

    def applyStep(self, action_id: int) -> Any:
        """Execute action_id through the ActionRegistry and update internal state."""
        raise NotImplementedError

    def addReward(self, amount: float, reason: str = "") -> None:
        """Accumulate reward for the current episode."""
        raise NotImplementedError

    def shapingPenalty(self) -> float:
        """Return any additional shaping penalty for the current step."""
        return 0.0

    @property
    def isWarmingUp(self) -> bool:
        """True during warmup/guided phase (no learning updates)."""
        return False

    # ── Cooperative stop handling ─────────────────────────────────────────────

    # ── Public action / step contract ────────────────────────────────────────

    def isOptionAction(self, semanticId: int) -> bool:
        """Return True if semanticId refers to a macro-action/option."""
        return False

    def executePrimitiveAction(self, semanticId: int) -> Any:
        """Execute a primitive action by semantic ID and return a StepResult."""
        raise NotImplementedError

    def selectAction(self, decision: Any) -> Any:
        """Choose an action given a DecisionInput; return ActionChoice."""
        raise NotImplementedError

    # ── Runtime status ────────────────────────────────────────────────────────

    def isStopRequested(self) -> bool:
        """Return True if a cooperative stop has been requested."""
        return self._stopRequested

    def getStatus(self) -> "BotStatus":
        """Return a normalized snapshot of this bot's runtime state for UI display."""
        raise NotImplementedError

    # ── Manual epsilon override ───────────────────────────────────────────────

    def setManualEpsilon(self, value: float | None) -> bool:
        """Override the exploration rate. Pass None to clear. Returns False if unsupported."""
        return False

    def persistTrainingArtifacts(self) -> None:
        """Save algorithm-specific training artifacts (Q-table, checkpoint, etc.).

        Called by PostEpisodeRecorder after each training episode.
        Subclasses override to save whatever the algorithm requires.
        """
        pass

    def getTrainingEpisodePersistencePolicy(self) -> EpisodePersistencePolicy:
        """Return the persistence policy for ordinary training episodes."""
        return EpisodePersistencePolicy(
            save_maze_episode=True,
            save_heatmap_stats=True,
            append_reward=True,
        )

    def getEvaluationEpisodeDefinition(self) -> EvaluationEpisodeDefinition:
        """Return the bot-owned definition of its evaluation episode."""
        return EvaluationEpisodeDefinition(
            description="This bot does not define a dedicated evaluation episode.",
            frequency=0,
            is_dedicated_episode=False,
            eval_epsilon=None,
            persistence=EpisodePersistencePolicy(),
        )

    def getEpisodePersistencePolicy(self, mode: str) -> EpisodePersistencePolicy:
        """Resolve persistence policy for a completed episode mode."""
        if mode == "training":
            return self.getTrainingEpisodePersistencePolicy()
        if mode == "evaluation":
            return self.getEvaluationEpisodeDefinition().persistence
        return EpisodePersistencePolicy()

    def shouldRunEvaluationEpisode(self, completedTrainingEpisodes: int) -> bool:
        """Return True when a dedicated evaluation episode should run."""
        definition = self.getEvaluationEpisodeDefinition()
        frequency = max(0, int(definition.frequency))
        return bool(
            definition.is_dedicated_episode
            and frequency > 0
            and completedTrainingEpisodes > 0
            and completedTrainingEpisodes % frequency == 0
        )

    def requestStop(self) -> None:
        self._stopRequested = True

    def clearStop(self) -> None:
        self._stopRequested = False
