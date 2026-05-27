from .bot import QLearning, QLearningBot
from .config import BOT_SPEC, QLearningConfig

BOT_TYPE = "QLearningBot"
BOT_CLASS = QLearningBot

__all__ = [
    "BOT_CLASS",
    "BOT_SPEC",
    "BOT_TYPE",
    "QLearning",
    "QLearningBot",
    "QLearningConfig",
]
