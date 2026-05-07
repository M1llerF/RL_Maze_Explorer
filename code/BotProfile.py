from __future__ import annotations

import os
import pickle
import tempfile
from typing import Any, cast

from BotConfigs import QLearningConfig
from RewardSystem import RewardConfig
from BotStatistics import BotStatistics



class BotProfile:
    def __init__(
        self,
        name: str,
        bot_type: str,
        config: QLearningConfig,
        reward_config: RewardConfig,
        statistics: BotStatistics,
        bot_specific_data: dict[str, Any],
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
        self.bot_type = bot_type
        self.config = config
        self.reward_config = reward_config
        self.statistics = statistics
        self.bot_specific_data = bot_specific_data

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the profile to a dictionary.

        :return: A dictionary representation of the profile.
        """
        config_data = self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        reward_config_data = self.reward_config.__dict__ if hasattr(self.reward_config, '__dict__') else self.reward_config
        statistics_data = self.statistics.__dict__ if hasattr(self.statistics, '__dict__') else self.statistics
        return {
            "name": self.name,
            "bot_type": self.bot_type,
            "config": config_data,
            "reward_config": reward_config_data,
            "statistics": statistics_data,
            "bot_specific_data": self.bot_specific_data
        }
    
    @staticmethod
    def from_dict(data: Any, default_name: str | None = None) -> BotProfile:
        """
        Create a BotProfile instance from a dictionary.

        :param data: A dictionary containing the profile data.
        :return: A BotProfile instance.
        """
        # Tolerant loader: handle missing keys and older profile schemas.
        d = dict(data or {})
        name = d.get('name') or default_name or 'Unnamed'

        # Infer bot type if missing
        bot_type = d.get('bot_type')
        cfg_raw = d.get('config')
        cfg_dict: dict[str, Any] = cast(dict[str, Any], cfg_raw) if isinstance(cfg_raw, dict) else {}
        if bot_type is None:
            bot_type = 'QLearningBot'

        # Build config safely with only known keys
        config_class = QLearningConfig
        allowed = {'learning_rate','discount_factor','use_position_in_state'}
        cfg_kwargs: dict[str, Any] = {}
        cfg_kwargs = {str(k): v for k, v in cfg_dict.items() if str(k) in allowed}
        try:
            config = config_class(**cfg_kwargs)
        except TypeError:
            config = config_class()

        # Reward config
        reward_config = d.get('reward_config')
        if isinstance(reward_config, dict):
            reward_config = RewardConfig(**cast(dict[str, Any], reward_config))
        elif not isinstance(reward_config, RewardConfig):
            reward_config = RewardConfig()

        # Statistics
        statistics = d.get('statistics')
        if isinstance(statistics, dict):
            s = BotStatistics()
            try:
                s.__dict__.update(cast(dict[str, Any], statistics))
            except Exception:
                pass
            statistics = s
        elif not isinstance(statistics, BotStatistics):
            statistics = BotStatistics()

        raw_specific = d.get('bot_specific_data')
        bot_specific_data: dict[str, Any] = (
            cast(dict[str, Any], raw_specific) if isinstance(raw_specific, dict) else {}
        )

        return BotProfile(
            name=name,
            bot_type=bot_type,
            config=config,
            reward_config=reward_config,
            statistics=statistics,
            bot_specific_data=bot_specific_data
        )

class ProfileManager:
    def __init__(self, profile_directory: str) -> None:
        """
        Initialize the ProfileManager with a directory for storing profiles.

        :param profile_directory: The directory where profiles are stored.
        """
        self.profile_directory = profile_directory

    def save_profile(self, profile: BotProfile) -> None:
        """
        Save a profile to a pickle file and create necessary files.

        :param profile: The BotProfile instance to save.
        """
        profile_dir = f"{self.profile_directory}/{profile.name}"
        os.makedirs(profile_dir, exist_ok=True)
        filename = f"{profile_dir}/profile.pkl"
        
        # Atomic write to avoid partial reads by other threads
        profile_dict = profile.to_dict()
        dir_path = os.path.dirname(filename)
        os.makedirs(dir_path, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=dir_path, mode='wb') as tmp:
            pickle.dump(profile_dict, tmp)
            temp_name = tmp.name
        os.replace(temp_name, filename)

        # Avoid creating empty q_table.pkl to prevent EOFError on first load.
        self._create_empty_file(os.path.join(profile_dir, "SimulationRewards.txt"))
        self._create_empty_file(os.path.join(profile_dir, "HeatmapData.txt"))


    def load_profile(self, profile_name: str) -> BotProfile:
        """
        Load a profile from a pickle file.

        :param profile_name: The name of the profile to load.
        :return: A BotProfile instance.
        """
        profile_dir = f"{self.profile_directory}/{profile_name}"
        filename = f"{profile_dir}/profile.pkl"

        # Handle occasional concurrent-write races gracefully
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        except EOFError:
            # If a write was in progress, retry once
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        return BotProfile.from_dict(data, default_name=profile_name)
    
    def list_profiles(self) -> list[str]:
        """
        List all available profiles.

        :return: A list of profile names.
        """
        return [d for d in os.listdir(self.profile_directory) if os.path.isdir(os.path.join(self.profile_directory, d))]

    @staticmethod
    def _create_empty_file(filepath: str) -> None:
        """
        Create a empty file if it doesn't exist.

        :param filepath: The path to the file to create.
        """
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write("")  # Write a empty string to create the file
