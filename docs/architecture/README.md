# RL Maze Explorer Architecture

This guide explains what is core/shared versus bot-specific, and what must change when you add a new bot.

## High-Level Shape
- UI layer (`code/ui/*`) drives user actions.
- Orchestration layer (`code/gameEnvironment.py`, `code/services/training.py`) manages profiles, bots, and training loops.
- Domain/infrastructure (`code/maze.py`, `code/rewardSystem.py`, `code/services/repository.py`) provides maze dynamics, rewards, and persistence.
- Bot implementations live under `code/bots/*`.

## What Is Bot-Specific
- `code/bots/<botName>/bot.py`: behavior and algorithm integration.
- `code/bots/<botName>/config.py`: config class + `BOT_SPEC`.
- `code/bots/<botName>/__init__.py`: exports bot class/config/spec.

## What Usually Stays Unchanged
- `code/services/repository.py` (artifact storage)
- `code/maze.py` and `code/pathfinding.py`
- Most of `code/ui/*` unless you need bot-specific controls

## Required Touchpoints For A New Bot
1. Add a new package under `code/bots/` with `BOT_SPEC`.
2. Export `BOT_TYPE` and `BOT_CLASS` in `code/bots/<botName>/__init__.py` for auto-registration.
3. Ensure the bot exposes the methods expected by runners/UI (`runEpisode`, `reset`, etc.).

## Diagrams
- `systemContext.puml`: ownership and boundaries.
- `componentMap.puml`: key modules and dependencies.
- `trainingSequence.puml`: one training episode flow.
- `addBotChecklist.puml`: operational flow to integrate a new bot.
