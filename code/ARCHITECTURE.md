# Code Organization

## Shared/Core (usually unchanged when adding a new bot)
- `gameEnvironment.py`: profile lifecycle, maze resets, bot instance management.
- `maze.py`, `pathfinding.py`: maze model and path calculations.
- `services/`: persistence (`repository.py`), episode runner (`runners.py`), training orchestration (`training.py`).
- `ui/`: app shell and frames.
- `botFactory.py`: constructs bots from registered types.
- `botProfile.py`, `botStatistics.py`: profile payload and runtime stats.
- `rewardSystem.py`: reward evaluation engine.

## Bot-Specific
- `bots/qlearning/bot.py`: Q-learning algorithm + bot behavior.

## Adding a New Bot
1. Create a new package under `bots/` (for example `bots/dqn/`).
2. Implement bot class and algorithm there.
3. Register the bot in `GameEnvironment.registerBots`.
4. Add bot config/spec in that bot package (for example `bots/<botName>/config.py` with `BOT_SPEC`).
5. Import bots by package path (for example `from bots.qlearning import QLearningBot`).
