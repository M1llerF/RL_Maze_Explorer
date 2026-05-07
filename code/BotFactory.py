from __future__ import annotations

from typing import Any, Protocol

from RewardSystem import RewardSystem, MazeSensors
from services.repository import ArtifactsRepository


class BotProtocol(Protocol):
    def initialize_specific_data(self, data: dict[str, Any]) -> None: ...


class BotFactory:
    def __init__(self, maze: Any, repository: ArtifactsRepository | None = None) -> None:
        """
        Initialize the BotFactory with a given maze.
        
        :param maze: The maze instance that the bots will navigate.
        """
        self.maze = maze
        self.bot_registry: dict[str, Any] = {}
        self.repository = repository or ArtifactsRepository()

    def register_bot(self, bot_type: str, bot_class: Any) -> None:
        """
        Register a new bot type with its corresponding class.

        :param bot_type: A string representing the type of the bot.
        :param bot_class: The class of the bot to be registered.
        """
        self.bot_registry[bot_type] = bot_class

    def create_bot(
        self,
        bot_type: str,
        profile_name: str,
        config: Any,
        reward_config: Any,
        statistics: Any,
        bot_specific_data: dict[str, Any],
    ) -> Any:
        """
        Create a instance of the specified bot type.

        :param bot_type: The type of bot to create.
        :param profile_name: The profile name for the bot.
        :param config: The configuration for the bot.
        :param reward_config: The reward configuration for the bot.
        :param statistics: The statistics instance to track the bot's performance.
        :param bot_specific_data: Specific data needed for the bot's initialization.

        :return: a instance of the specified bot.
        
        :raises ValueError: If the bot type is not registered.
        """
        if bot_type not in self.bot_registry:
            raise ValueError(f"Unknown bot type: {bot_type}")

        bot_class = self.bot_registry[bot_type]
        # Provide minimal sensors interface decoupled from BotTools
        reward_system = RewardSystem(self.maze, reward_config, sensors=MazeSensors(self.maze))
        # Inject shared repository instance
        try:
            bot_instance = bot_class(
                self.maze,
                config,
                reward_system,
                statistics,
                profile_name=profile_name,
                repository=self.repository,
            )
        except TypeError:
            # Backward compatibility if constructor doesn't accept repository yet
            bot_instance = bot_class(self.maze, config, reward_system, statistics, profile_name=profile_name)
        
        if hasattr(bot_instance, 'initialize_specific_data'):
            bot_instance.initialize_specific_data(bot_specific_data)

        return bot_instance
