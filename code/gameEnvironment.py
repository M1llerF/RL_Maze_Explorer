from botFactory import BotFactory
from maze import Maze
from botStatistics import BotStatistics
from botProfile import BotProfile, ProfileManager
from typing import Any, Optional, cast
import threading
import time
from services.repository import ArtifactsRepository
import os

class GameEnvironment:
    def __init__(self, width: int = 10, height: int = 10, profile_directory: str = 'profiles'):
        """
        Initialize the GameEnvironment with a maze, bot factory, and profile manager.

        :param width: Width of the maze.
        :param height: Height of the maze.
        :param profile_directory: Directory where profiles are stored.
        """
        self.maze = Maze(width, height)
        # Single shared repository instance for all artifacts
        try:
            os.makedirs(profile_directory, exist_ok=True)
        except Exception:
            pass
        self.repository = ArtifactsRepository(profile_directory)
        self.bot_factory = BotFactory(self.maze, repository=self.repository)
        self.profile_manager = ProfileManager(profile_directory)
        self.bots: list[Any] = []
        self.register_bots()
        # Training pause control per profile
        self._pause_lock = threading.Lock()
        self._paused_profiles: set[str] = set()
        # Episode completion tracking per profile (used by training UI)
        self._completed_lock = threading.Lock()
        self._completed_episodes: dict[str, int] = {}
        # Training maze pool (fixed MDP per phase)
        self.training_pool_active: bool = False
        self.training_pool: list[dict[str, Any]] = []
        self.training_pool_index: int = 0
        # Optional fixed custom maze (overrides random/pool when active)
        self.fixed_maze_active: bool = False
        self.fixed_maze_state: Optional[dict[str, Any]] = None

    # ----- Fixed custom maze controls ----
    def set_fixed_maze(self, state: dict[str, Any]) -> None:
        """Enable and set a fixed custom maze state to use on each reset."""
        self.fixed_maze_state = state
        self.fixed_maze_active = True
        # Disable training pool when using fixed maze
        self.training_pool_active = False

    def clear_fixed_maze(self) -> None:
        """Disable fixed custom maze usage."""
        self.fixed_maze_state = None
        self.fixed_maze_active = False

    def pause_training_for(self, profile_name: str) -> None:
        with self._pause_lock:
            self._paused_profiles.add(profile_name)

    def resume_training_for(self, profile_name: str) -> None:
        with self._pause_lock:
            self._paused_profiles.discard(profile_name)

    def _is_paused(self, profile_name: str) -> bool:
        with self._pause_lock:
            return profile_name in self._paused_profiles

    # Public helper for UI to check paused state
    def is_training_paused(self, profile_name: str) -> bool:
        return self._is_paused(profile_name)

    # Completed episode tracking APIs
    def reset_completed(self, profile_name: str) -> None:
        with self._completed_lock:
            self._completed_episodes[profile_name] = 0

    def inc_completed(self, profile_name: str) -> None:
        with self._completed_lock:
            self._completed_episodes[profile_name] = self._completed_episodes.get(profile_name, 0) + 1

    def get_completed(self, profile_name: str) -> int:
        with self._completed_lock:
            return self._completed_episodes.get(profile_name, 0)

    def register_bots(self) -> None:
        """
        Register available bots with the bot factory.
        """
        from qLearningBot import QLearningBot  # Ensure QLearningBot is imported only when needed
        self.bot_factory.register_bot('QLearningBot', QLearningBot)
        # Register other bots as needed
        # self.bot_factory.register_bot('AnotherBot', AnotherBot)
        # Additional bots can be registered here
        
    def setup_new_profile(self, profile_name: str, bot_type: str, config: Any, reward_config: Any) -> None:
        """
        Set up a new bot profile and save it.

        :param profile_name: Name of the profile.
        :param bot_type: Type of the bot.
        :param config: Configuration for the bot.
        :param reward_config: Reward configuration for the bot.
        """
        profile = BotProfile(profile_name, bot_type, config, reward_config, BotStatistics(), {})
        self.profile_manager.save_profile(profile)
        # Only attempt to instantiate a bot if the type is registered
        try:
            if bot_type in self.bot_factory.bot_registry:
                self.setup_bots(profile.bot_type, profile.name, config, reward_config, profile.statistics, profile.bot_specific_data)
        except Exception:
            # Ignore bot creation failures during profile setup so profiles can still be created
            pass

    def game_loop(self, rounds: int, bot_index: int, visualize: bool = False, visualization_window: Optional[Any] = None) -> None:
        """
        Run the game loop for a specified number of rounds.

        :param rounds: Number of rounds to run.
        :param bot_index: Index of the bot to run.
        :param visualize: Whether to visualize the game.
        :param visualization_window: Visualization window object.
        """
        bot = self.bots[bot_index]
        for _ in range(rounds):
            # If visualization is open for this profile, pause training thread
            while self._is_paused(bot.profile_name):
                time.sleep(0.05)
            bot.run_episode()
            self.reset_environment(bot_index)
            # Track completion for training UI
            self.inc_completed(bot.profile_name)
            if visualize and visualization_window:
                visualization_window.update_visualization()

    def configure_training_pool(self, size: int = 20) -> None:
        """Build a fixed pool of random mazes to cycle through during training."""
        self.training_pool = []
        for _ in range(max(1, size)):
            self.maze.setup_simple_maze()
            self.training_pool.append(self.maze.get_state())
        self.training_pool_index = 0
        self.training_pool_active = True
        # Pool overrides any fixed maze selection
        self.fixed_maze_active = False

    def reset_environment(self, bot_index: int) -> None:
        """
        Reset the environment for the specified bot.

        :param bot_index: Index of the bot to reset.
        """
        bot = self.bots[bot_index]
        # Use training pool during training; keep maze stable in visualization unless a pool is configured
        if self._is_paused(bot.profile_name):
            # Visualization is active for this profile
            if self.fixed_maze_active and self.fixed_maze_state is not None:
                cast(Any, self.maze).set_state(self.fixed_maze_state)
            else:
                self.maze.setup_simple_maze()
        elif self.training_pool_active and self.training_pool:
            state = self.training_pool[self.training_pool_index]
            cast(Any, self.maze).set_state(state)
            self.training_pool_index = (self.training_pool_index + 1) % len(self.training_pool)
        else:
            if self.fixed_maze_active and self.fixed_maze_state is not None:
                cast(Any, self.maze).set_state(self.fixed_maze_state)
            else:
                self.maze.setup_simple_maze()
        for bot in self.bots:
            if bot == self.bots[bot_index]:
                # Use standardized reset interface
                try:
                    bot.reset()
                except AttributeError:
                    # Backward-compat if a bot still exposes reset_bot
                    if hasattr(bot, 'reset_bot'):
                        bot.reset_bot()

    def load_profile(self, profile_name: str) -> None:
        """
        Load a bot profile from the profile manager.

        :param profile_name: Name of the profile to load.
        """
        profile = self.profile_manager.load_profile(profile_name)
        self.apply_profile(profile)

    def apply_profile(self, profile: BotProfile) -> int:
        """
        Apply a loaded profile to the environment.

        :param profile: The bot profile to apply.
        :return: The index of the bot.
        """
        # Check if a bot with the same profile name already exists
        bot_index = next((i for i, bot in enumerate(self.bots) if bot.profile_name == profile.name), -1)
        if bot_index == -1:
            # If the bot does not exist, create a new one and append it
            bot = self.bot_factory.create_bot(
                profile.bot_type,
                profile.name,
                profile.config,
                profile.reward_config,
                profile.statistics,
                profile.bot_specific_data
            )
            self.bots.append(bot)
            bot_index = len(self.bots) - 1
        else:
            bot = self.bots[bot_index]
            bot.config = profile.config
            bot.reward_config = profile.reward_config
            bot.statistics = profile.statistics
            bot.bot_specific_data = profile.bot_specific_data

        return bot_index

    def setup_bots(
        self,
        bot_type: str,
        bot_name: str,
        config: Any,
        reward_config: Any,
        statistics: Any,
        bot_specific_data: dict[str, Any],
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
        self.bots.append(self.bot_factory.create_bot(bot_type, bot_name, config, reward_config, statistics, bot_specific_data))

    def save_profiles(self) -> None:
        """
        Save all bot profiles to the profile manager.
        """
        for bot in self.bots:
            profile = BotProfile(
                name = bot.profile_name,
                bot_type=type(bot).__name__,
                config=bot.config,
                reward_config=bot.reward_system.reward_config,
                bot_specific_data=bot.get_bot_specific_data(),
                statistics=bot.statistics
            )
            self.profile_manager.save_profile(profile)
