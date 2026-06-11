from __future__ import annotations

import os
import shutil

from botProfile import BotProfile, ProfileManager
from botStatistics import BotStatistics
from services.repository import ArtifactsRepository


class ProfileService:
    """Coordinates ProfileManager (config) and ArtifactsRepository (artifacts).

    ProfileManager owns profile identity and config (profile.pkl).
    ArtifactsRepository owns training outputs (q-table, rewards, stats, mazes).
    Neither should coordinate the other — that is this class's job.
    """

    def __init__(self, profileManager: ProfileManager, repository: ArtifactsRepository) -> None:
        self.profileManager = profileManager
        self.repository = repository

    def initializeProfile(self, profile: BotProfile) -> None:
        """Save a new profile config and create its artifact stubs."""
        self.profileManager.saveProfile(profile)
        self.repository.ensureProfileArtifacts(profile.name)

    def renameProfile(self, oldName: str, newName: str) -> None:
        """Rename a profile directory without discarding its training artifacts."""
        oldName = str(oldName).strip()
        newName = str(newName).strip()
        if not oldName or not newName or oldName == newName:
            return
        oldDir = os.path.join(self.profileManager.profileDirectory, oldName)
        newDir = os.path.join(self.profileManager.profileDirectory, newName)
        if not os.path.isdir(oldDir):
            raise FileNotFoundError(oldName)
        if os.path.exists(newDir):
            raise FileExistsError(newName)
        shutil.move(oldDir, newDir)

    def resetTrainingState(self, profileName: str) -> BotProfile:
        """Clear training artifacts and reset statistics, returning the updated profile.

        Raises FileNotFoundError if the profile does not exist.
        """
        profile = self.profileManager.loadProfile(profileName)
        profile.statistics = BotStatistics()
        profile.botSpecificData = {
            key: value
            for key, value in dict(profile.botSpecificData).items()
            if str(key).startswith("linked")
        }
        self.repository.clearProfileTrainingArtifacts(profileName)
        self.profileManager.saveProfile(profile)
        return profile

    def deleteProfile(self, profileName: str) -> None:
        """Remove all data for a profile (config + artifacts)."""
        profile_dir = os.path.join(self.profileManager.profileDirectory, profileName)
        if os.path.exists(profile_dir):
            shutil.rmtree(profile_dir)
        # Legacy flat .pkl path — present in very old installs
        legacy_pkl = os.path.join(self.profileManager.profileDirectory, f"{profileName}.pkl")
        if os.path.exists(legacy_pkl):
            os.remove(legacy_pkl)
