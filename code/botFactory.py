from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rewardSystem import RewardSystem, MazeSensors
from environment.traversal import build_traversal_policy
from services.diagnostics import DiagnosticsService
from services.repository import ArtifactsRepository


@dataclass
class BotCreateContext:
    """Single stable construction contract for all bot types.

    BotFactory creates one of these and passes it as the sole constructor
    argument to every bot class. No per-bot constructor signature filtering.
    """

    maze: Any
    config: Any
    rewardSystem: Any
    statistics: Any
    profileName: str
    repository: ArtifactsRepository
    botSpecificData: dict[str, Any] = field(default_factory=lambda: {})
    loadCheckpoint: bool = True
    diagnostics: DiagnosticsService | None = None

    @property
    def reward_system(self) -> Any:
        return self.rewardSystem

    @property
    def profile_name(self) -> str:
        return self.profileName

    @property
    def bot_specific_data(self) -> dict[str, Any]:
        return self.botSpecificData

    @property
    def load_checkpoint(self) -> bool:
        return self.loadCheckpoint




class BotFactory:
    def __init__(
        self,
        maze: Any,
        repository: ArtifactsRepository | None = None,
        diagnostics: DiagnosticsService | None = None,
    ) -> None:
        """
        Initialize the BotFactory with a given maze.
        
        :param maze: The maze instance that the bots will navigate.
        """
        self.maze = maze
        self.botRegistry: dict[str, Any] = {}
        self.repository = repository or ArtifactsRepository()
        self.diagnostics = diagnostics

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
        loadCheckpoint: bool = True,
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
        traversal = build_traversal_policy(self.maze, profileName)
        rewardSystem = RewardSystem(
            self.maze,
            rewardConfig,
            sensors=MazeSensors(self.maze, traversal=traversal),
            traversal=traversal,
        )
        ctx = BotCreateContext(
            maze=self.maze,
            config=config,
            rewardSystem=rewardSystem,
            statistics=statistics,
            profileName=profileName,
            repository=self.repository,
            botSpecificData=botSpecificData,
            loadCheckpoint=loadCheckpoint,
            diagnostics=self.diagnostics,
        )
        botInstance = botClass(ctx)

        if hasattr(botInstance, 'initializeSpecificData'):
            botInstance.initializeSpecificData(botSpecificData)

        return botInstance
