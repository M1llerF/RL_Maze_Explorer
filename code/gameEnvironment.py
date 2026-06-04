from botFactory import BotFactory
from maze import Maze
from botStatistics import BotStatistics
from botProfile import BotProfile, ProfileManager
from typing import Any, Optional
import time
from services.repository import ArtifactsRepository
from services.profile_service import ProfileService
from services.linked_profile_service import LinkedProfileService
from services.bot_runtime_manager import BotRuntimeManager
from services.diagnostics import DiagnosticsService
from services.maze_provider import MazeProvider
from services.warmup_service import WarmupService
import os

class GameEnvironment:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        profileDirectory: str = 'profiles',
        diagnostics: DiagnosticsService | None = None,
    ):
        """
        Initialize the GameEnvironment with a maze, bot factory, and profile manager.

        :param width: Width of the maze.
        :param height: Height of the maze.
        :param profile_directory: Directory where profiles are stored.
        """
        self.maze = Maze(width, height)
        self.diagnostics = diagnostics
        # Single shared repository instance for all artifacts
        try:
            os.makedirs(profileDirectory, exist_ok=True)
        except Exception:
            pass
        self.repository = ArtifactsRepository(profileDirectory)
        self.botFactory = BotFactory(
            self.maze,
            repository=self.repository,
            diagnostics=self.diagnostics,
        )
        self.profileManager = ProfileManager(profileDirectory)
        self.profileService = ProfileService(self.profileManager, self.repository)
        self.linkedProfileService = LinkedProfileService(
            self.profileManager,
            self.profileService,
            diagnostics=self.diagnostics,
        )
        self.warmupService = WarmupService(self.profileManager, self.repository)
        self.botRuntime = BotRuntimeManager(self.botFactory)
        self.botRuntime.registerBotTypes()
        self.mazeProvider = MazeProvider(self.maze)

    @property
    def bots(self) -> list[Any]:
        """Live bot list owned by BotRuntimeManager. Property keeps external callers unchanged."""
        return self.botRuntime.bots

    @property
    def trainingPoolActive(self) -> bool:
        return self.mazeProvider.trainingPoolActive

    @trainingPoolActive.setter
    def trainingPoolActive(self, value: bool) -> None:
        self.mazeProvider.trainingPoolActive = bool(value)

    @property
    def trainingPool(self) -> list[dict[str, Any]]:
        return self.mazeProvider.trainingPool

    @trainingPool.setter
    def trainingPool(self, value: list[dict[str, Any]]) -> None:
        self.mazeProvider.trainingPool = value

    @property
    def trainingPoolIndex(self) -> int:
        return self.mazeProvider.trainingPoolIndex

    @trainingPoolIndex.setter
    def trainingPoolIndex(self, value: int) -> None:
        self.mazeProvider.trainingPoolIndex = int(value)

    @property
    def fixedMazeActive(self) -> bool:
        return self.mazeProvider.fixedMazeActive

    @fixedMazeActive.setter
    def fixedMazeActive(self, value: bool) -> None:
        self.mazeProvider.fixedMazeActive = bool(value)

    @property
    def fixedMazeState(self) -> dict[str, Any] | None:
        return self.mazeProvider.fixedMazeState

    @fixedMazeState.setter
    def fixedMazeState(self, value: dict[str, Any] | None) -> None:
        self.mazeProvider.fixedMazeState = value

    @property
    def curriculumActive(self) -> bool:
        return self.mazeProvider.curriculumActive

    @curriculumActive.setter
    def curriculumActive(self, value: bool) -> None:
        self.mazeProvider.curriculumActive = bool(value)

    def hasFixedMazeConfigured(self) -> bool:
        return self.mazeProvider.fixedMazeActive and self.mazeProvider.fixedMazeState is not None

    # ----- Maze lifecycle delegates → MazeProvider ----
    def setFixedMaze(self, state: dict[str, Any]) -> None:
        self.mazeProvider.setFixedMaze(state)

    def clearFixedMaze(self) -> None:
        self.mazeProvider.clearFixedMaze()

    def setRandomGenerationLengthRange(
        self,
        minLength: int | None,
        maxLength: int | None,
    ) -> None:
        self.mazeProvider.setRandomGenerationLengthRange(minLength, maxLength)

    def _setupRandomMaze(self) -> None:
        self.mazeProvider.setupRandomMaze()

    def configureMazeMode(
        self,
        mode: str,
        *,
        poolSize: int = 20,
        minLength: int | None = None,
        maxLength: int | None = None,
    ) -> None:
        if mode == "Fixed (Builder)" and not self.hasFixedMazeConfigured():
            raise RuntimeError("Fixed maze is not configured.")
        self.mazeProvider.configureMode(
            mode,
            poolSize=poolSize,
            minLength=minLength,
            maxLength=maxLength,
        )

    def pauseTrainingFor(self, profileName: str) -> None:
        self.botRuntime.pause(profileName)

    def resumeTrainingFor(self, profileName: str) -> None:
        self.botRuntime.resume(profileName)

    def _isPaused(self, profileName: str) -> bool:
        return self.botRuntime.isPaused(profileName)

    def isTrainingPaused(self, profileName: str) -> bool:
        return self.botRuntime.isPaused(profileName)

    def resetCompleted(self, profileName: str) -> None:
        self.botRuntime.resetCompleted(profileName)

    def incCompleted(self, profileName: str) -> None:
        self.botRuntime.incCompleted(profileName)

    def getCompleted(self, profileName: str) -> int:
        return self.botRuntime.getCompleted(profileName)
        
    def setupNewProfile(self, profileName: str, botType: str, config: Any, rewardConfig: Any) -> None:
        """
        Set up a new bot profile and save it.

        :param profile_name: Name of the profile.
        :param bot_type: Type of the bot.
        :param config: Configuration for the bot.
        :param reward_config: Reward configuration for the bot.
        """
        profile = BotProfile(profileName, botType, config, rewardConfig, BotStatistics(), {})
        self.profileService.initializeProfile(profile)
        self.linkedProfileService.syncLinkedProfile(profile)
        # Only attempt to instantiate a bot if the type is registered
        try:
            if botType in self.botFactory.botRegistry:
                self.setupBots(profile.botType, profile.name, config, rewardConfig, profile.statistics, profile.botSpecificData)
        except Exception as e:
            if self.diagnostics is not None:
                self.diagnostics.exception(
                    "game_environment",
                    "Bot creation failed after profile save",
                    e,
                    profile_name=profile.name,
                    bot_type=profile.botType,
                )

    def gameLoop(self, rounds: int, botIndex: int, visualize: bool = False, visualizationWindow: Optional[Any] = None) -> None:
        """
        Run the game loop for a specified number of rounds.

        :param rounds: Number of rounds to run.
        :param bot_index: Index of the bot to run.
        :param visualize: Whether to visualize the game.
        :param visualization_window: Visualization window object.
        """
        bot = self.bots[botIndex]
        for _ in range(rounds):
            # If visualization is open for this profile, pause training thread
            while self._isPaused(bot.profileName):
                time.sleep(0.05)
            bot.runEpisode()
            self.resetEnvironment(botIndex)
            # Track completion for training UI
            self.incCompleted(bot.profileName)
            if visualize and visualizationWindow:
                visualizationWindow.updateVisualization()

    def configureTrainingPool(self, size: int = 20) -> None:
        self.mazeProvider.configureTrainingPool(size)

    def getMazeSize(self) -> tuple[int, int]:
        return self.mazeProvider.getMazeSize()

    def setMazeSize(self, width: int, height: int) -> None:
        self.mazeProvider.setMazeSize(width, height)

    def resetEnvironment(self, botIndex: int) -> None:
        """
        Reset the environment for the specified bot.

        :param bot_index: Index of the bot to reset.
        """
        bot = self.bots[botIndex]
        self.mazeProvider.resetMaze(self._isPaused(bot.profileName))
        for bot in self.bots:
            if bot == self.bots[botIndex]:
                # Use standardized reset interface
                try:
                    bot.reset()
                except AttributeError:
                    # Backward-compat if a bot still exposes reset_bot
                    if hasattr(bot, 'reset_bot'):
                        bot.reset_bot()

    def loadProfile(self, profileName: str) -> None:
        """
        Load a bot profile from the profile manager.

        :param profile_name: Name of the profile to load.
        """
        profile = self.profileManager.loadProfile(profileName)
        self.applyProfile(profile)

    def applyProfile(self, profile: BotProfile, loadCheckpoint: bool = True) -> int:
        """
        Apply a loaded profile to the environment.

        :param profile: The bot profile to apply.
        :return: The index of the bot.
        """
        # Check if a bot with the same profile name already exists
        botIndex = next((i for i, bot in enumerate(self.bots) if bot.profileName == profile.name), -1)
        bot = self.botFactory.createBot(
            profile.botType,
            profile.name,
            profile.config,
            profile.rewardConfig,
            profile.statistics,
            profile.botSpecificData,
            loadCheckpoint=loadCheckpoint,
        )
        if botIndex == -1:
            # If the bot does not exist, create a new one and append it
            self.bots.append(bot)
            botIndex = len(self.bots) - 1
        else:
            # Replace in-place so callers keeping bot indices remain valid.
            self.bots[botIndex] = bot

        return botIndex

    def setupBots(
        self,
        botType: str,
        botName: str,
        config: Any,
        rewardConfig: Any,
        statistics: Any,
        botSpecificData: dict[str, Any],
    ) -> None:
        """
        Set up bots and add them to the environment.

        :param bot_type: Type of the bot.
        :param bot_name: Name of the bot.
        :param config: Configuration for the bot.
        :param reward_config: Reward configuration for the bot.
        :param statistics: Statistics object for the bot.
        :param bot_specific_data: Specific data for the bot.
        """
        self.bots.append(self.botFactory.createBot(botType, botName, config, rewardConfig, statistics, botSpecificData))

    def saveProfiles(self) -> None:
        """
        Save all bot profiles to the profile manager.
        """
        for bot in self.bots:
            profile = BotProfile(
                name = bot.profileName,
                botType=type(bot).__name__,
                config=bot.config,
                rewardConfig=bot.rewardSystem.rewardConfig,
                botSpecificData=bot.getBotSpecificData(),
                statistics=bot.statistics
            )
            self.profileManager.saveProfile(profile)
            self.linkedProfileService.syncLinkedProfile(profile)

    def resetProfileTrainingData(self, profileName: str) -> None:
        """Clear persisted training artifacts and reload the profile in an untrained state."""
        profile = self.profileService.resetTrainingState(profileName)
        self.linkedProfileService.syncLinkedProfile(profile)
        if profile.botType == "DQNBot":
            linkedProfile = self.linkedProfileService.resetLinkedProfile(profile.name, profile.config)
            if linkedProfile is not None:
                self.applyProfile(linkedProfile, loadCheckpoint=False)
        self.resetCompleted(profileName)
        self.resumeTrainingFor(profileName)
        self.applyProfile(profile, loadCheckpoint=False)
