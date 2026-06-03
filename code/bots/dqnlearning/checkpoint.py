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


class CheckpointIO:
    ARTIFACT_NAME = "dqn_model.pt"
    PAYLOAD_VERSION = 3

    def __init__(self, repo: ArtifactsRepository, profileName: str, device: torch.device) -> None:
        self.repo = repo
        self.profileName = profileName
        self.device = device

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
            },
            "replay": replayState,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        self.repo.saveModelArtifactBytes(self.profileName, self.ARTIFACT_NAME, buffer.getvalue())

    def load(self, expectedMeta: CheckpointMeta) -> dict[str, Any] | None:
        checkpoint, _ = self.loadWithReason(expectedMeta)
        return checkpoint

    def loadWithReason(self, expectedMeta: CheckpointMeta) -> tuple[dict[str, Any] | None, str]:
        raw = self.repo.loadModelArtifactBytes(self.profileName, self.ARTIFACT_NAME)
        if raw is None:
            return None, "checkpoint_missing"
        try:
            checkpoint = torch.load(io.BytesIO(raw), map_location=self.device)
            if int(checkpoint.get("version", 0)) != self.PAYLOAD_VERSION:
                return None, "payload_version_mismatch"
            meta = checkpoint.get("meta", {})
            loadedMeta = CheckpointMeta(
                stateSchemaVersion=int(meta.get("state_schema_version", -1)),
                encoderConfigFingerprint=str(meta.get("encoder_fingerprint", "")),
                inputDim=int(meta.get("input_dim", -1)),
            )
            if loadedMeta != expectedMeta:
                # Backward-compatible restore:
                # allow checkpoints when structural schema still matches
                # (same version + input dimensionality) but fingerprint changed
                # due to non-structural fingerprint fields across code revisions.
                if (
                    loadedMeta.stateSchemaVersion == expectedMeta.stateSchemaVersion
                    and loadedMeta.inputDim == expectedMeta.inputDim
                ):
                    return checkpoint, "ok_compat_fingerprint"
                return None, (
                    f"meta_mismatch loaded={loadedMeta} expected={expectedMeta}"
                )
            return checkpoint, "ok"
        except Exception:
            return None, "load_exception"
