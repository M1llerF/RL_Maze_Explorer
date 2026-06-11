from __future__ import annotations

from typing import Any, cast

from services.diagnostics import DiagnosticsService

from .agent import DqnAgent
from .checkpoint import CheckpointIO, CheckpointMeta
from .hrlAgent import HierarchicalDqnAgent
from .replay import UniformReplayStore


PRIMITIVE_LINKED_PROFILE_SUFFIX = "_primitive"
LEGACY_PRIMITIVE_LINKED_PROFILE_SUFFIX = "_primtive"


class DQNCheckpointService:
    """Owns DQN checkpoint save/load policy for a single profile."""

    def __init__(
        self,
        checkpoint: CheckpointIO,
        primitiveCheckpoint: CheckpointIO,
        checkpointMeta: CheckpointMeta,
        primitiveCheckpointMeta: CheckpointMeta,
        *,
        profileName: str,
        legacyPrimitiveCheckpoint: CheckpointIO | None = None,
        diagnostics: DiagnosticsService | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.primitiveCheckpoint = primitiveCheckpoint
        self.legacyPrimitiveCheckpoint = legacyPrimitiveCheckpoint
        self.checkpointMeta = checkpointMeta
        self.primitiveCheckpointMeta = primitiveCheckpointMeta
        self.profileName = profileName
        self._diagnostics = diagnostics

    def save(self, bot: Any) -> None:
        freq = int(getattr(bot.config, "checkpointFrequency", 10))
        if freq > 0 and int(bot.episodeCounter) % freq != 0:
            return

        extraState: dict[str, Any] = {}
        if isinstance(bot.agent, HierarchicalDqnAgent):
            extraState.update({
                "low_policy": bot.agent.lowLevel.policy.state_dict(),
                "low_target": bot.agent.lowLevel.target.state_dict(),
                "low_optimizer": bot.agent.lowLevel.optimizer.state_dict(),
            })
        manualEpsilonOverride = getattr(bot.agent, "manualEpsilonOverride", None)
        if manualEpsilonOverride is not None:
            extraState["manual_epsilon_override"] = float(manualEpsilonOverride)
        self.checkpoint.save(
            policyStateDict=bot.agent.policy.state_dict(),
            targetStateDict=bot.agent.target.state_dict(),
            optimizerStateDict=bot.agent.optimizer.state_dict(),
            globalStep=bot.agent.globalStep,
            episodeCount=bot.episodeCounter,
            meta=self.checkpointMeta,
            replayState=None,
            extraState=extraState or None,
        )

        if bot.primitiveBootstrapAgent is not None:
            self.primitiveCheckpoint.save(
                policyStateDict=bot.primitiveBootstrapAgent.policy.state_dict(),
                targetStateDict=bot.primitiveBootstrapAgent.target.state_dict(),
                optimizerStateDict=bot.primitiveBootstrapAgent.optimizer.state_dict(),
                globalStep=bot.primitiveBootstrapAgent.globalStep,
                episodeCount=bot.episodeCounter,
                meta=self.primitiveCheckpointMeta,
                replayState=None,
                extraState=None,
            )
            CheckpointIO(
                bot.repo,
                self._primitiveLinkedProfileName(),
                bot.agent.device,
            ).save(
                policyStateDict=bot.primitiveBootstrapAgent.policy.state_dict(),
                targetStateDict=bot.primitiveBootstrapAgent.target.state_dict(),
                optimizerStateDict=bot.primitiveBootstrapAgent.optimizer.state_dict(),
                globalStep=bot.primitiveBootstrapAgent.globalStep,
                episodeCount=bot.episodeCounter,
                meta=self.primitiveCheckpointMeta,
                replayState=None,
                extraState=None,
            )
            if self.legacyPrimitiveCheckpoint is not None:
                self.legacyPrimitiveCheckpoint.save(
                    policyStateDict=bot.primitiveBootstrapAgent.policy.state_dict(),
                    targetStateDict=bot.primitiveBootstrapAgent.target.state_dict(),
                    optimizerStateDict=bot.primitiveBootstrapAgent.optimizer.state_dict(),
                    globalStep=bot.primitiveBootstrapAgent.globalStep,
                    episodeCount=bot.episodeCounter,
                    meta=self.primitiveCheckpointMeta,
                    replayState=None,
                    extraState=None,
                )

    def load(self, bot: Any) -> bool:
        checkpoint, reason = self.checkpoint.loadWithReason(self.checkpointMeta)
        if checkpoint is None:
            if self._tryLoadPrimitiveBootstrapFallback(bot):
                return True
            self._log("info", "Checkpoint not loaded", reason=reason)
            return False
        try:
            bot.agent.policy.load_state_dict(checkpoint["policy"])
            bot.agent.target.load_state_dict(checkpoint["target"])
            bot.agent.optimizer.load_state_dict(checkpoint["optimizer"])
            if isinstance(bot.agent, HierarchicalDqnAgent):
                extra = cast(dict[str, Any], checkpoint.get("extra", {}))
                bot.agent.lowLevel.policy.load_state_dict(extra["low_policy"])
                bot.agent.lowLevel.target.load_state_dict(extra["low_target"])
                bot.agent.lowLevel.optimizer.load_state_dict(extra["low_optimizer"])
            elif bot.primitiveBootstrapAgent is not None:
                self._loadPrimitiveBootstrapSidecar(bot, checkpoint)
        except Exception as exc:
            self._log("warning", "Checkpoint state mismatch", error=str(exc))
            return False

        bot.agent.globalStep = int(checkpoint.get("global_step", 0))
        bot.episodeCounter = int(checkpoint.get("episode_count", 0))
        extra = checkpoint.get("extra")
        if isinstance(extra, dict):
            manualEpsilonOverride = extra.get("manual_epsilon_override")
            if manualEpsilonOverride is not None:
                setter = getattr(bot.agent, "setManualEpsilon", None)
                if callable(setter):
                    setter(float(manualEpsilonOverride))
                else:
                    bot.agent.manualEpsilonOverride = float(manualEpsilonOverride)
        replayState = checkpoint.get("replay")
        if replayState is not None and isinstance(bot.agent.replay, UniformReplayStore):
            bot.agent.replay.load_serializable(cast(dict[str, Any], replayState))
        compatibilityNote = " [compat]" if reason == "ok_compat_fingerprint" else ""
        self._log(
            "info",
            f"Checkpoint loaded{compatibilityNote}",
            global_step=int(bot.agent.globalStep),
            replay_size=len(bot.agent.replay),
        )
        return True

    def _tryLoadPrimitiveBootstrapFallback(self, bot: Any) -> bool:
        if bot.usesHierarchicalPolicy or bot.usesMacroOnlyPolicy or getattr(bot.config, "useMacroActions", False):
            return False
        for linkedProfileName in (
            self._primitiveLinkedProfileName(),
            self._legacyPrimitiveLinkedProfileName(),
        ):
            linkedPrimitiveCheckpoint = CheckpointIO(
                bot.repo,
                linkedProfileName,
                bot.agent.device,
            ).load(self.primitiveCheckpointMeta)
            if linkedPrimitiveCheckpoint is None:
                continue
            try:
                bot.agent.policy.load_state_dict(linkedPrimitiveCheckpoint["policy"])
                bot.agent.target.load_state_dict(linkedPrimitiveCheckpoint["target"])
                bot.agent.optimizer.load_state_dict(linkedPrimitiveCheckpoint["optimizer"])
            except Exception:
                continue
            bot.agent.globalStep = int(linkedPrimitiveCheckpoint.get("global_step", 0))
            self._log(
                "info",
                "Primitive bootstrap loaded",
                global_step=int(bot.agent.globalStep),
                load_source="linked_profile",
                linked_profile_name=linkedProfileName,
            )
            return True
        primitiveCheckpoint = self.primitiveCheckpoint.load(self.primitiveCheckpointMeta)
        if primitiveCheckpoint is not None:
            try:
                bot.agent.policy.load_state_dict(primitiveCheckpoint["policy"])
                bot.agent.target.load_state_dict(primitiveCheckpoint["target"])
                bot.agent.optimizer.load_state_dict(primitiveCheckpoint["optimizer"])
            except Exception:
                primitiveCheckpoint = None
            else:
                bot.agent.globalStep = int(primitiveCheckpoint.get("global_step", 0))
                self._log(
                    "info",
                    "Primitive bootstrap loaded",
                    global_step=int(bot.agent.globalStep),
                    load_source="sidecar",
                )
                return True
        if self.legacyPrimitiveCheckpoint is not None:
            primitiveCheckpoint = self.legacyPrimitiveCheckpoint.load(self.primitiveCheckpointMeta)
            if primitiveCheckpoint is not None:
                try:
                    bot.agent.policy.load_state_dict(primitiveCheckpoint["policy"])
                    bot.agent.target.load_state_dict(primitiveCheckpoint["target"])
                    bot.agent.optimizer.load_state_dict(primitiveCheckpoint["optimizer"])
                except Exception:
                    primitiveCheckpoint = None
                else:
                    bot.agent.globalStep = int(primitiveCheckpoint.get("global_step", 0))
                    self._log(
                        "info",
                        "Primitive bootstrap loaded",
                        global_step=int(bot.agent.globalStep),
                        load_source="legacy_sidecar",
                    )
                    return True

        rawCheckpoint = self.checkpoint.loadRaw()
        if rawCheckpoint is None:
            return False
        rawExtra = rawCheckpoint.get("extra")
        if not isinstance(rawExtra, dict):
            return False
        extra = cast(dict[str, Any], rawExtra)
        rawMeta = extra.get("primitive_bootstrap_meta")
        if not isinstance(rawMeta, dict):
            return False
        meta = cast(dict[str, Any], rawMeta)
        expectedActionDim = bot._primitiveActionSpace().numActions
        if int(meta.get("state_schema_version", -1)) != int(self.checkpointMeta.stateSchemaVersion):
            return False
        if int(meta.get("input_dim", -1)) != int(self.checkpointMeta.inputDim):
            return False
        if int(meta.get("action_dim", -1)) != int(expectedActionDim):
            return False
        try:
            bot.agent.policy.load_state_dict(extra["primitive_bootstrap_policy"])
            bot.agent.target.load_state_dict(extra["primitive_bootstrap_target"])
            bot.agent.optimizer.load_state_dict(extra["primitive_bootstrap_optimizer"])
        except Exception:
            return False
        bot.agent.globalStep = int(extra.get("primitive_bootstrap_global_step", 0))
        self._log(
            "info",
            "Primitive bootstrap loaded",
            global_step=int(bot.agent.globalStep),
            load_source="embedded",
        )
        return True

    def _loadPrimitiveBootstrapSidecar(self, bot: Any, parentCheckpoint: dict[str, Any]) -> None:
        if bot.primitiveBootstrapAgent is None:
            return
        primitiveCheckpoint = self.primitiveCheckpoint.load(self.primitiveCheckpointMeta)
        if primitiveCheckpoint is not None:
            bot.primitiveBootstrapAgent.policy.load_state_dict(primitiveCheckpoint["policy"])
            bot.primitiveBootstrapAgent.target.load_state_dict(primitiveCheckpoint["target"])
            bot.primitiveBootstrapAgent.optimizer.load_state_dict(primitiveCheckpoint["optimizer"])
            bot.primitiveBootstrapAgent.globalStep = int(primitiveCheckpoint.get("global_step", 0))
            return
        rawExtra = parentCheckpoint.get("extra")
        if not isinstance(rawExtra, dict):
            return
        extra = cast(dict[str, Any], rawExtra)
        if "primitive_bootstrap_policy" not in extra:
            return
        bot.primitiveBootstrapAgent.policy.load_state_dict(extra["primitive_bootstrap_policy"])
        bot.primitiveBootstrapAgent.target.load_state_dict(extra["primitive_bootstrap_target"])
        bot.primitiveBootstrapAgent.optimizer.load_state_dict(extra["primitive_bootstrap_optimizer"])
        bot.primitiveBootstrapAgent.globalStep = int(extra.get("primitive_bootstrap_global_step", 0))

    def _primitiveLinkedProfileName(self) -> str:
        return f"{self.profileName}{PRIMITIVE_LINKED_PROFILE_SUFFIX}"

    def _legacyPrimitiveLinkedProfileName(self) -> str:
        return f"{self.profileName}{LEGACY_PRIMITIVE_LINKED_PROFILE_SUFFIX}"

    def _log(self, level: str, message: str, **context: Any) -> None:
        if self._diagnostics is None:
            return
        self._diagnostics.log(
            level,
            "dqn_checkpoint_service",
            message,
            profile_name=self.profileName,
            **context,
        )
