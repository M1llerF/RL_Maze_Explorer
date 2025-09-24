class BaseBot:
    def __init__(self, maze, statistics, config=None):
        """
        Initialize the base bot.

        :param maze: The maze object that the bot will navigate.
        :param statistics: a instance of BotStatistics to track the bot's performance.
        :param config: Optional configuration for specific algorithms (e.g., Q-learning config).
        """
        self.maze = maze
        self.statistics = statistics
        self.config = config
        # Cooperative stop flag to abort long episodes promptly
        self._stop_requested = False
    
    def reset(self):
        """Reset the bot's state and statistics. Should be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def calculate_state(self):
        """Calculate the current state of the bot. Should be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def run_episode(self):
        """Run a single episode of the bot's operation. Should be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    # Cooperative stop handling
    def request_stop(self):
        self._stop_requested = True

    def clear_stop(self):
        self._stop_requested = False
