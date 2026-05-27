from botFactory import BotFactory
from maze import Maze
from botStatistics import BotStatistics
from botProfile import BotProfile, ProfileManager
from typing import Any, Optional, cast
import threading
import time
from services.repository import ArtifactsRepository
from bots import discoverBotClasses
import os

class GameEnvironment:
    def __init__(self, width: int = 10, height: int = 10, profileDirectory: str = 'profiles'):
        """
        Initialize the GameEnvironment with a maze, bot factory, and profile manager.

        :param width: Width of the maze.
        :param height: Height of the maze.
        :param profile_directory: Directory where profiles are stored.
        """
        self.maze = Maze(width, height)
        # Single shared repository instance for all artifacts
        try:
            os.makedirs(profileDirectory, exist_ok=True)
        except Exception:
            pass
        self.repository = ArtifactsRepository(profileDirectory)
        self.botFactory = BotFactory(self.maze, repository=self.repository)
        self.profileManager = ProfileManager(profileDirectory)
        self.bots: list[Any] = []
        self.registerBots()
        # Training pause control per profile
        self._pauseLock = threading.Lock()
        self._pausedProfiles: set[str] = set()
        # Episode completion tracking per profile (used by training UI)
        self._completedLock = threading.Lock()
        self._completedEpisodes: dict[str, int] = {}
        # Training maze pool (fixed MDP per phase)
        self.trainingPoolActive: bool = False
        self.trainingPool: list[dict[str, Any]] = []
        self.trainingPoolIndex: int = 0
        # Optional fixed custom maze (overrides random/pool when active)
        self.fixedMazeActive: bool = False
        self.fixedMazeState: Optional[dict[str, Any]] = None

    # ----- Fixed custom maze controls ----
    def setFixedMaze(self, state: dict[str, Any]) -> None:
        """Enable and set a fixed custom maze state to use on each reset."""
        self.fixedMazeState = state
        self.fixedMazeActive = True
        # Disable training pool when using fixed maze
        self.trainingPoolActive = False

    def clearFixedMaze(self) -> None:
        """Disable fixed custom maze usage."""
        self.fixedMazeState = None
        self.fixedMazeActive = False

    def pauseTrainingFor(self, profileName: str) -> None:
        with self._pauseLock:
            self._pausedProfiles.add(profileName)

    def resumeTrainingFor(self, profileName: str) -> None:
        with self._pauseLock:
            self._pausedProfiles.discard(profileName)

    def _isPaused(self, profileName: str) -> bool:
        with self._pauseLock:
            return profileName in self._pausedProfiles

    # Public helper for UI to check paused state
    def isTrainingPaused(self, profileName: str) -> bool:
        return self._isPaused(profileName)

    # Completed episode tracking APIs
    def resetCompleted(self, profileName: str) -> None:
        with self._completedLock:
            self._completedEpisodes[profileName] = 0

    def incCompleted(self, profileName: str) -> None:
        with self._completedLock:
            self._completedEpisodes[profileName] = self._completedEpisodes.get(profileName, 0) + 1

    def getCompleted(self, profileName: str) -> int:
        with self._completedLock:
            return self._completedEpisodes.get(profileName, 0)

    def registerBots(self) -> None:
        """
        Register available bots with the bot factory.
        """
        for botType, botClass in discoverBotClasses().items():
            self.botFactory.registerBot(botType, botClass)
        
    def setupNewProfile(self, profileName: str, botType: str, config: Any, rewardConfig: Any) -> None:
        """
        Set up a new bot profile and save it.

        :param profile_name: Name of the profile.
        :param bot_type: Type of the bot.
        :param config: Configuration for the bot.
        :param reward_config: Reward configuration for the bot.
        """
        profile = BotProfile(profileName, botType, config, rewardConfig, BotStatistics(), {})
        self.profileManager.saveProfile(profile)
        # Only attempt to instantiate a bot if the type is registered
        try:
            if botType in self.botFactory.botRegistry:
                self.setupBots(profile.botType, profile.name, config, rewardConfig, profile.statistics, profile.botSpecificData)
        except Exception:
            # Ignore bot creation failures during profile setup so profiles can still be created
            pass

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
        """Build a fixed pool of random mazes to cycle through during training."""
        self.trainingPool = []
        for _ in range(max(1, size)):
            self.maze.setupSimpleMaze()
            self.trainingPool.append(self.maze.getState())
        self.trainingPoolIndex = 0
        self.trainingPoolActive = True
        # Pool overrides any fixed maze selection
        self.fixedMazeActive = False

    def resetEnvironment(self, botIndex: int) -> None:
        """
        Reset the environment for the specified bot.

        :param bot_index: Index of the bot to reset.
        """
        bot = self.bots[botIndex]
        # Use training pool during training; keep maze stable in visualization unless a pool is configured
        if self._isPaused(bot.profileName):
            # Visualization is active for this profile
            if self.fixedMazeActive and self.fixedMazeState is not None:
                cast(Any, self.maze).setState(self.fixedMazeState)
            else:
                self.maze.setupSimpleMaze()
        elif self.trainingPoolActive and self.trainingPool:
            state = self.trainingPool[self.trainingPoolIndex]
            cast(Any, self.maze).setState(state)
            self.trainingPoolIndex = (self.trainingPoolIndex + 1) % len(self.trainingPool)
        else:
            if self.fixedMazeActive and self.fixedMazeState is not None:
                cast(Any, self.maze).setState(self.fixedMazeState)
            else:
                self.maze.setupSimpleMaze()
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

    def applyProfile(self, profile: BotProfile) -> int:
        """
        Apply a loaded profile to the environment.

        :param profile: The bot profile to apply.
        :return: The index of the bot.
        """
        # Check if a bot with the same profile name already exists
        botIndex = next((i for i, bot in enumerate(self.bots) if bot.profileName == profile.name), -1)
        if botIndex == -1:
            # If the bot does not exist, create a new one and append it
            bot = self.botFactory.createBot(
                profile.botType,
                profile.name,
                profile.config,
                profile.rewardConfig,
                profile.statistics,
                profile.botSpecificData
            )
            self.bots.append(bot)
            botIndex = len(self.bots) - 1
        else:
            bot = self.bots[botIndex]
            bot.config = profile.config
            bot.rewardConfig = profile.rewardConfig
            bot.statistics = profile.statistics
            bot.botSpecificData = profile.botSpecificData

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
