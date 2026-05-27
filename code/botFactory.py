from __future__ import annotations

import inspect
from typing import Any, Protocol

from rewardSystem import RewardSystem, MazeSensors
from services.repository import ArtifactsRepository


class BotProtocol(Protocol):
    def initializeSpecificData(self, data: dict[str, Any]) -> None: ...


class BotFactory:
    def __init__(self, maze: Any, repository: ArtifactsRepository | None = None) -> None:
        """
        Initialize the BotFactory with a given maze.
        
        :param maze: The maze instance that the bots will navigate.
        """
        self.maze = maze
        self.botRegistry: dict[str, Any] = {}
        self.repository = repository or ArtifactsRepository()

    def registerBot(self, botType: str, botClass: Any) -> None:
        """
        Register a new bot type with its corresponding class.

        :param bot_type: A string representing the type of the bot.
        :param bot_class: The class of the bot to be registered.
        """
        self.botRegistry[botType] = botClass

    def isRegistered(self, botType: str) -> bool:
        """Return whether the given bot type is registered."""
        return botType in self.botRegistry

    def listRegisteredBotTypes(self) -> list[str]:
        """Return sorted registered bot type names for diagnostics/UI callers."""
        return sorted(self.botRegistry.keys())

    def createBot(
        self,
        botType: str,
        profileName: str,
        config: Any,
        rewardConfig: Any,
        statistics: Any,
        botSpecificData: dict[str, Any],
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
        if not self.isRegistered(botType):
            known = self.listRegisteredBotTypes()
            knownMsg = ", ".join(known) if known else "<none>"
            raise ValueError(f"Unknown bot type: {botType}. Registered bot types: {knownMsg}")

        botClass = self.botRegistry[botType]
        # Provide minimal sensors interface decoupled from BotTools
        rewardSystem = RewardSystem(self.maze, rewardConfig, sensors=MazeSensors(self.maze))
        # Constructor argument filtering keeps multi-bot extension straightforward
        # without per-bot branching in the factory.
        constructorArgs: dict[str, Any] = {
            "maze": self.maze,
            "config": config,
            "rewardSystem": rewardSystem,
            "reward_system": rewardSystem,
            "statistics": statistics,
            "profileName": profileName,
            "profile_name": profileName,
            "repository": self.repository,
        }
        signature = inspect.signature(botClass)
        acceptsKwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if acceptsKwargs:
            ctorKwargs = constructorArgs
        else:
            accepted = {
                name
                for name, p in signature.parameters.items()
                if name != "self"
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            ctorKwargs = {k: v for k, v in constructorArgs.items() if k in accepted}
        botInstance = botClass(**ctorKwargs)
        
        if hasattr(botInstance, 'initializeSpecificData'):
            botInstance.initializeSpecificData(botSpecificData)
        elif hasattr(botInstance, 'initialize_specific_data'):
            botInstance.initialize_specific_data(botSpecificData)

        return botInstance
