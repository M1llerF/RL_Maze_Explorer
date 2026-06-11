from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from botProfile import ProfileManager
from services.repository import ArtifactsRepository


@dataclass
class WarmupStatusResult:
    """View model returned to UI for warmup compatibility state.

    When is_compatible is False, status_message explains why.
    When is_compatible is True with no options, status_message says so.
    When is_compatible is True with options, status_message is empty and
    the caller formats the message from the selected option.
    """
    is_compatible: bool
    status_message: str
    options: list[Any] = field(default_factory=lambda: [])  # list[WarmupStoreOption]


class WarmupService:
    """Service layer for warmup compatibility checks and store management.

    Wraps WarmupStore so UI frames do not import DQN-specific code directly.
    WarmupStore is imported lazily to avoid loading torch at startup.
    """

    def __init__(self, profileManager: ProfileManager, repository: ArtifactsRepository) -> None:
        self.profileManager = profileManager
        self.repository = repository

    def getWarmupStatus(self, profileName: str) -> WarmupStatusResult:
        """Return warmup compatibility and available options for a profile."""
        from bots.dqnlearning.warmupStore import WarmupStore

        if not profileName:
            return WarmupStatusResult(
                is_compatible=False,
                status_message="Select a compatible profile to use warmup features",
            )

        try:
            profileObj = self.profileManager.loadProfile(profileName)
        except Exception:
            return WarmupStatusResult(
                is_compatible=False,
                status_message="Select a compatible profile to use warmup features",
            )

        compatibility = WarmupStore.inferCompatibilityForProfile(profileObj)
        if compatibility is None:
            botType = str(getattr(profileObj, "botType", "selected bot"))
            return WarmupStatusResult(
                is_compatible=False,
                status_message=f"Warmup unavailable for {botType}",
            )

        fingerprint, inputDim, actionDim = compatibility
        profileNames = list(self.profileManager.listProfiles())
        options = WarmupStore.loadCompatibleOptions(
            self.repository,
            profileNames,
            selectedProfile=profileName,
            fingerprint=fingerprint,
            inputDim=inputDim,
            actionDim=actionDim,
        )

        if not options:
            return WarmupStatusResult(
                is_compatible=True,
                status_message="Warmup supported. No compatible warmup stores saved yet.",
                options=[],
            )

        return WarmupStatusResult(is_compatible=True, status_message="", options=options)

    def hasStore(self, profileName: str) -> bool:
        """Return True if a warmup store artifact exists for the profile."""
        from bots.dqnlearning.warmupStore import WarmupStore
        return WarmupStore.exists(self.repository, profileName)

    def deleteStore(self, profileName: str) -> None:
        """Delete the warmup store artifact for the profile, if present."""
        from bots.dqnlearning.warmupStore import WarmupStore
        import os
        path = os.path.join(self.repository.modelArtifactsDir(profileName), WarmupStore.ARTIFACT_NAME)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
