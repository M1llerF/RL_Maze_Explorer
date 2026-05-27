from __future__ import annotations

import inspect
from typing import Any, Protocol

from rewardSystem import RewardSystem, MazeSensors
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

    def is_registered(self, bot_type: str) -> bool:
        """Return whether the given bot type is registered."""
        return bot_type in self.bot_registry

    def list_registered_bot_types(self) -> list[str]:
        """Return sorted registered bot type names for diagnostics/UI callers."""
        return sorted(self.bot_registry.keys())

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
        if not self.is_registered(bot_type):
            known = self.list_registered_bot_types()
            known_msg = ", ".join(known) if known else "<none>"
            raise ValueError(f"Unknown bot type: {bot_type}. Registered bot types: {known_msg}")

        bot_class = self.bot_registry[bot_type]
        # Provide minimal sensors interface decoupled from BotTools
        reward_system = RewardSystem(self.maze, reward_config, sensors=MazeSensors(self.maze))
        # Constructor argument filtering keeps multi-bot extension straightforward
        # without per-bot branching in the factory.
        constructor_args: dict[str, Any] = {
            "maze": self.maze,
            "config": config,
            "reward_system": reward_system,
            "statistics": statistics,
            "profile_name": profile_name,
            "repository": self.repository,
        }
        signature = inspect.signature(bot_class)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if accepts_kwargs:
            ctor_kwargs = constructor_args
        else:
            accepted = {
                name
                for name, p in signature.parameters.items()
                if name != "self"
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            ctor_kwargs = {k: v for k, v in constructor_args.items() if k in accepted}
        bot_instance = bot_class(**ctor_kwargs)
        
        if hasattr(bot_instance, 'initialize_specific_data'):
            bot_instance.initialize_specific_data(bot_specific_data)

        return bot_instance
