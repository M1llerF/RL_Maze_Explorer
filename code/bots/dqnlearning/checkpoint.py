from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Any

import torch

from services.repository import ArtifactsRepository


@dataclass(frozen=True)
class CheckpointMeta:
    stateSchemaVersion: int
    encoderConfigFingerprint: str
    inputDim: int
    actionDim: int = 0
    policyMode: str = "flat"


class CheckpointIO:
    ARTIFACT_NAME = "dqn_model.pt"
    PAYLOAD_VERSION = 4

    def __init__(
        self,
        repo: ArtifactsRepository,
        profileName: str,
        device: torch.device,
        artifactName: str | None = None,
    ) -> None:
        self.repo = repo
        self.profileName = profileName
        self.device = device
        self.artifactName = artifactName or self.ARTIFACT_NAME

    def save(
        self,
        *,
        policyStateDict: dict[str, Any],
        targetStateDict: dict[str, Any],
        optimizerStateDict: dict[str, Any],
        globalStep: int,
        episodeCount: int,
        meta: CheckpointMeta,
        replayState: dict[str, Any] | None = None,
        extraState: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "version": self.PAYLOAD_VERSION,
            "policy": policyStateDict,
            "target": targetStateDict,
            "optimizer": optimizerStateDict,
            "global_step": int(globalStep),
            "episode_count": int(episodeCount),
            "meta": {
                "state_schema_version": int(meta.stateSchemaVersion),
                "encoder_fingerprint": str(meta.encoderConfigFingerprint),
                "input_dim": int(meta.inputDim),
                "action_dim": int(meta.actionDim),
                "policy_mode": str(meta.policyMode),
            },
            "replay": replayState,
            "extra": extraState,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        self.repo.saveModelArtifactBytes(self.profileName, self.artifactName, buffer.getvalue())

    def load(self, expectedMeta: CheckpointMeta) -> dict[str, Any] | None:
        checkpoint, _ = self.loadWithReason(expectedMeta)
        return checkpoint

    def loadRaw(self) -> dict[str, Any] | None:
        raw = self.repo.loadModelArtifactBytes(self.profileName, self.artifactName)
        if raw is None:
            return None
        try:
            checkpoint = torch.load(io.BytesIO(raw), map_location=self.device)
            if int(checkpoint.get("version", 0)) != self.PAYLOAD_VERSION:
                return None
            return checkpoint
        except Exception:
            return None

    def loadWithReason(self, expectedMeta: CheckpointMeta) -> tuple[dict[str, Any] | None, str]:
        checkpoint = self.loadRaw()
        if checkpoint is None:
            return None, "checkpoint_missing"
        try:
            meta = checkpoint.get("meta", {})
            loadedMeta = CheckpointMeta(
                stateSchemaVersion=int(meta.get("state_schema_version", -1)),
                encoderConfigFingerprint=str(meta.get("encoder_fingerprint", "")),
                inputDim=int(meta.get("input_dim", -1)),
                actionDim=int(meta.get("action_dim", 0)),
                policyMode=str(meta.get("policy_mode", "flat")),
            )
            if loadedMeta != expectedMeta:
                # Backward-compatible restore:
                # allow checkpoints when structural schema still matches
                # (same version + input dimensionality) but fingerprint changed
                # due to non-structural fingerprint fields across code revisions.
                if (
                    loadedMeta.stateSchemaVersion == expectedMeta.stateSchemaVersion
                    and loadedMeta.inputDim == expectedMeta.inputDim
                    and loadedMeta.actionDim == expectedMeta.actionDim
                    and loadedMeta.policyMode == expectedMeta.policyMode
                ):
                    return checkpoint, "ok_compat_fingerprint"
                return None, (
                    f"meta_mismatch loaded={loadedMeta} expected={expectedMeta}"
                )
            return checkpoint, "ok"
        except Exception:
            return None, "load_exception"
