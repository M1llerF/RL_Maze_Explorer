from __future__ import annotations

import os
import pickle
import tempfile
from typing import Any, cast

from botConfigs import buildConfigForBotType
from rewardSystem import RewardConfig
from botStatistics import BotStatistics



class BotProfile:
    def __init__(
        self,
        name: str,
        botType: str,
        config: Any,
        rewardConfig: RewardConfig,
        statistics: BotStatistics,
        botSpecificData: dict[str, Any],
    ) -> None:
        """
        Initialize the BotProfile with the provided parameters.

        :param name: The name of the bot profile.
        :param bot_type: The type of the bot.
        :param config: The configuration object for the bot.
        :param reward_config: The reward configuration object for the bot.
        :param statistics: The statistics object tracking the bot's performance.
        :param bot_specific_data: Additional data specific to the bot.
        """
        self.name = name
        self.botType = botType
        self.config = config
        self.rewardConfig = rewardConfig
        self.statistics = statistics
        self.botSpecificData = botSpecificData

    def toDict(self) -> dict[str, Any]:
        """
        Convert the profile to a dictionary.

        :return: A dictionary representation of the profile.
        """
        configData = self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        rewardConfigData = self.rewardConfig.__dict__ if hasattr(self.rewardConfig, '__dict__') else self.rewardConfig
        statisticsData = self.statistics.__dict__ if hasattr(self.statistics, '__dict__') else self.statistics
        return {
            "name": self.name,
            "bot_type": self.botType,
            "config": configData,
            "reward_config": rewardConfigData,
            "statistics": statisticsData,
            "bot_specific_data": self.botSpecificData
        }
    
    @staticmethod
    def _legacySnakeToCamel(d: dict[str, Any]) -> dict[str, Any]:
        out = dict(d)
        for key, value in d.items():
            if "_" in key:
                parts = [p for p in key.split("_") if p]
                if parts:
                    camel = parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])
                    out.setdefault(camel, value)
        return out

    @staticmethod
    def fromDict(data: Any, defaultName: str | None = None) -> BotProfile:
        """
        Create a BotProfile instance from a dictionary.

        :param data: A dictionary containing the profile data.
        :return: A BotProfile instance.
        """
        # Tolerant loader: handle missing keys and older profile schemas.
        d = dict(data or {})
        name = d.get('name') or defaultName or 'Unnamed'

        # Infer bot type if missing
        botType = d.get('bot_type')
        cfgRaw = d.get('config')
        if botType is None:
            botType = 'QLearningBot'

        # Build config using bot-type mapping; defaults to QLearning for legacy payloads.
        config = buildConfigForBotType(botType, cfgRaw)

        # Reward config
        rewardConfig = d.get('reward_config')
        if isinstance(rewardConfig, dict):
            rewardConfig = RewardConfig(**BotProfile._legacySnakeToCamel(cast(dict[str, Any], rewardConfig)))
        elif not isinstance(rewardConfig, RewardConfig):
            rewardConfig = RewardConfig()

        # Statistics
        statistics = d.get('statistics')
        if isinstance(statistics, dict):
            s = BotStatistics()
            try:
                s.__dict__.update(BotProfile._legacySnakeToCamel(cast(dict[str, Any], statistics)))
            except Exception:
                pass
            statistics = s
        elif not isinstance(statistics, BotStatistics):
            statistics = BotStatistics()

        rawSpecific = d.get('bot_specific_data')
        botSpecificData: dict[str, Any] = (
            cast(dict[str, Any], rawSpecific) if isinstance(rawSpecific, dict) else {}
        )

        return BotProfile(
            name=name,
            botType=botType,
            config=config,
            rewardConfig=rewardConfig,
            statistics=statistics,
            botSpecificData=botSpecificData
        )

class ProfileManager:
    def __init__(self, profileDirectory: str) -> None:
        """
        Initialize the ProfileManager with a directory for storing profiles.

        :param profile_directory: The directory where profiles are stored.
        """
        self.profileDirectory = profileDirectory

    def saveProfile(self, profile: BotProfile) -> None:
        """
        Save a profile to a pickle file and create necessary files.

        :param profile: The BotProfile instance to save.
        """
        profileDir = f"{self.profileDirectory}/{profile.name}"
        os.makedirs(profileDir, exist_ok=True)
        filename = f"{profileDir}/profile.pkl"
        
        # Atomic write to avoid partial reads by other threads
        profileDict = profile.toDict()
        dirPath = os.path.dirname(filename)
        os.makedirs(dirPath, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=dirPath, mode='wb') as tmp:
            pickle.dump(profileDict, tmp)
            tempName = tmp.name
        os.replace(tempName, filename)

        # Avoid creating empty q_table.pkl to prevent EOFError on first load.
        self._createEmptyFile(os.path.join(profileDir, "SimulationRewards.txt"))
        self._createEmptyFile(os.path.join(profileDir, "HeatmapData.txt"))


    def loadProfile(self, profileName: str) -> BotProfile:
        """
        Load a profile from a pickle file.

        :param profile_name: The name of the profile to load.
        :return: A BotProfile instance.
        """
        profileDir = f"{self.profileDirectory}/{profileName}"
        filename = f"{profileDir}/profile.pkl"

        # Handle occasional concurrent-write races gracefully
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        except EOFError:
            # If a write was in progress, retry once
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        return BotProfile.fromDict(data, defaultName=profileName)
    
    def listProfiles(self) -> list[str]:
        """
        List all available profiles.

        :return: A list of profile names.
        """
        return [d for d in os.listdir(self.profileDirectory) if os.path.isdir(os.path.join(self.profileDirectory, d))]

    @staticmethod
    def _createEmptyFile(filepath: str) -> None:
        """
        Create a empty file if it doesn't exist.

        :param filepath: The path to the file to create.
        """
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write("")  # Write a empty string to create the file
