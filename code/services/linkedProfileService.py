from __future__ import annotations

from dataclasses import replace, is_dataclass
from typing import Any

from botProfile import BotProfile, ProfileManager
from botStatistics import BotStatistics
from services.diagnostics import DiagnosticsService
from services.profileService import ProfileService


class LinkedProfileService:
    """Manages the primitive linked-profile side policy for DQN bots.

    When dualRecordPrimitivePolicy is enabled, a DQN bot trains a second
    profile that records only primitive actions. This service owns the
    creation and synchronisation of that linked profile so GameEnvironment
    does not need to know about it.
    """

    PRIMITIVE_SUFFIX = "_primitive"
    LEGACY_PRIMITIVE_SUFFIX = "_primtive"

    def __init__(
        self,
        profileManager: ProfileManager,
        profileService: ProfileService,
        diagnostics: DiagnosticsService | None = None,
    ) -> None:
        self.profileManager = profileManager
        self.profileService = profileService
        self._diagnostics = diagnostics

    @classmethod
    def linked_profile_name(cls, profileName: str) -> str:
        return f"{profileName}{cls.PRIMITIVE_SUFFIX}"

    @classmethod
    def legacy_linked_profile_name(cls, profileName: str) -> str:
        return f"{profileName}{cls.LEGACY_PRIMITIVE_SUFFIX}"

    def syncLinkedProfile(self, profile: BotProfile) -> None:
        """Create or update the primitive-side linked profile for a DQN profile."""
        if profile.botType != "DQNBot":
            return
        if not getattr(profile.config, "dualRecordPrimitivePolicy", False):
            return

        linkedName = self.linked_profile_name(profile.name)
        try:
            existing = self._loadExistingLinkedProfile(profile.name)
            statistics = existing.statistics
            botSpecificData = dict(existing.botSpecificData)
        except FileNotFoundError:
            statistics = BotStatistics()
            botSpecificData = {}
        except Exception as e:
            if self._diagnostics is not None:
                self._diagnostics.exception(
                    "linked_profile_service",
                    "Could not load linked profile for synchronization",
                    e,
                    profile_name=profile.name,
                    linked_profile_name=linkedName,
                )
            statistics = BotStatistics()
            botSpecificData = {}

        primitiveConfig = self.primitiveLinkedConfig(profile.config)
        botSpecificData.update({
            "linkedParentProfile": profile.name,
            "linkedPrimitiveProfile": True,
        })
        linkedProfile = BotProfile(
            linkedName,
            profile.botType,
            primitiveConfig,
            profile.rewardConfig,
            statistics,
            botSpecificData,
        )
        self.profileManager.saveProfile(linkedProfile)

    def renameLinkedProfiles(self, oldProfileName: str, newProfileName: str) -> None:
        if oldProfileName == newProfileName:
            return
        renamePairs = (
            (self.linked_profile_name(oldProfileName), self.linked_profile_name(newProfileName)),
            (self.legacy_linked_profile_name(oldProfileName), self.legacy_linked_profile_name(newProfileName)),
        )
        for oldLinkedName, newLinkedName in renamePairs:
            try:
                self.profileService.renameProfile(oldLinkedName, newLinkedName)
            except FileNotFoundError:
                continue

    def resetLinkedProfile(self, profileName: str, profileConfig: Any) -> BotProfile | None:
        """Clear training state for the linked profile; return the profile or None if absent."""
        if not getattr(profileConfig, "dualRecordPrimitivePolicy", False):
            return None
        candidateNames = (
            self.linked_profile_name(profileName),
            self.legacy_linked_profile_name(profileName),
        )
        for linkedName in candidateNames:
            try:
                return self.profileService.resetTrainingState(linkedName)
            except FileNotFoundError:
                continue
            except Exception as e:
                if self._diagnostics is not None:
                    self._diagnostics.exception(
                        "linked_profile_service",
                        "Could not reset linked profile",
                        e,
                        profile_name=profileName,
                        linked_profile_name=linkedName,
                    )
                return None
        return None

    def _loadExistingLinkedProfile(self, profileName: str) -> BotProfile:
        for linkedName in (
            self.linked_profile_name(profileName),
            self.legacy_linked_profile_name(profileName),
        ):
            try:
                return self.profileManager.loadProfile(linkedName)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(profileName)

    @staticmethod
    def primitiveLinkedConfig(config: Any) -> Any:
        """Return a config with macro/hierarchical flags stripped for the primitive side policy."""
        if is_dataclass(config) and not isinstance(config, type):
            return replace(  # pyright: ignore[reportUnknownVariableType,reportArgumentType]
                config,
                useMacroActions=False,
                useMacroOnlyPolicy=False,
                dualRecordPrimitivePolicy=False,
                useHierarchicalPolicy=False,
            )
        return config
