import argparse
import json
import os
from typing import Any, Dict, List, cast
from services.repository import ArtifactsRepository

from gameEnvironment import GameEnvironment
from botConfigs import QLearningConfig, botConfigs
from rewardSystem import RewardConfig


def _readProfileStats(profileName: str) -> Dict[str, Any]:
    """Read aggregated profile stats via repository, returning defaults when absent."""
    try:
        repo = ArtifactsRepository("profiles")
        return repo.readProfileStats(profileName) or {}
    except Exception:
        return {}


def _delta(after: Dict[str, Any], before: Dict[str, Any], key: str) -> int:
    return int(after.get(key, 0) - before.get(key, 0))


def runProfileEpisodes(profileName: str, episodes: int) -> Dict[str, Any]:
    """
    Run a profile for N episodes without GUI and collect debug metrics.

    Returns a dictionary with per-episode metrics and summary stats.
    """
    env: Any = GameEnvironment()

    # Load profile and create/apply bot
    profile = env.profileManager.loadProfile(profileName)
    botIndex = int(env.applyProfile(profile))
    bot: Any = env.bots[botIndex]

    # Static reference values
    optimalPath: list[tuple[int, int]] = cast(
        list[tuple[int, int]],
        bot.tools.getOptimalPathInfo(bot.maze.start, bot.maze.end, output="path"),
    )
    optimalLen = len(optimalPath)

    perEpisode: List[Dict[str, Any]] = []

    for ep in range(episodes):
        before = _readProfileStats(profileName)

        bot.runEpisode()

        after = _readProfileStats(profileName)

        success = bot.position == bot.maze.end

        stepsTotal = _delta(after, before, "total_steps")
        stepsUnique = _delta(after, before, "non_repeating_steps_taken")
        stepsRevisited = _delta(after, before, "times_revisited_squares")
        wallsHit = _delta(after, before, "times_hit_wall")

        perEpisode.append({
            "episode": ep + 1,
            "reward": float(bot.totalReward),
            "success": bool(success),
            "steps_total": int(stepsTotal),
            "steps_unique": int(stepsUnique),
            "steps_revisited": int(stepsRevisited),
            "walls_hit": int(wallsHit),
            "q_table_states": int(len(bot.qLearning.qTable)),
            "unique_vs_optimal": float(stepsUnique / optimalLen) if optimalLen > 0 else 0.0,
        })

        # Reset environment for next episode
        env.resetEnvironment(botIndex)

    # Summaries
    rewards = [e["reward"] for e in perEpisode]
    successes = [1 if e["success"] else 0 for e in perEpisode]
    qSizes = [e["q_table_states"] for e in perEpisode]
    walls = [e["walls_hit"] for e in perEpisode]

    summary = {
        "episodes": episodes,
        "profile": profileName,
        "success_rate": sum(successes) / episodes if episodes else 0.0,
        "avg_reward": sum(rewards) / episodes if episodes else 0.0,
        "avg_walls_hit": sum(walls) / episodes if episodes else 0.0,
        "qtable_growth": qSizes[-1] - qSizes[0] if episodes > 1 else 0,
        "optimal_path_len": optimalLen,
    }
    successRate = float(summary["success_rate"])
    avgReward = float(summary["avg_reward"])
    avgWallsHit = float(summary["avg_walls_hit"])
    qtableGrowth = int(summary["qtable_growth"])

    # Simple heuristics for failure hints
    hints: List[str] = []
    if successRate < 0.25 and avgReward < 0:
        hints.append("Low success and negative rewards: agent likely not reaching goal; consider lowering exploration decay or adjusting rewards.")
    if avgWallsHit > 5:
        hints.append("High wall collisions: increase wall penalty or improve state features.")
    if qtableGrowth <= 0 and episodes > 5:
        hints.append("Q-table not growing: exploration may be too low or episodes too short.")

    return {"summary": summary, "per_episode": perEpisode, "hints": hints}


def createProfile(profileName: str, botType: str = "QLearningBot", lr: float = 0.1, gamma: float = 0.9,
                   usePositionInState: bool = True, potential: bool = False, progressScale: float = 5.0) -> None:
    """Create a new profile directory with default config and reward settings."""
    env: Any = GameEnvironment()

    if botType != "QLearningBot":
        raise ValueError("Currently only QLearningBot is supported by this CLI.")

    # Build config
    config = QLearningConfig(learningRate=lr, discountFactor=gamma, usePositionInState=usePositionInState)

    # Build reward config from bot_configs defaults
    rewards = botConfigs.get(botType, {}).get("rewards", {})
    rewardConfig = RewardConfig(rewardModifiers=rewards, usePotentialShaping=potential, progressScale=progressScale)

    env.setupNewProfile(profileName, botType, config, rewardConfig)
    print(f"Created profile '{profileName}' for {botType} (lr={lr}, gamma={gamma}, pos_state={usePositionInState}, potential={potential}, progress_scale={progressScale}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or create profiles for headless training and metrics.")
    parser.add_argument("--profile", required=True, help="Profile name (folder under profiles/)")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes to run")
    parser.add_argument("--save-json", action="store_true", help="Save results to profiles/<profile>/debug_results.json")
    parser.add_argument("--create", action="store_true", help="Create a new profile before running")
    parser.add_argument("--bot", default="QLearningBot", help="Bot type to create (default: QLearningBot)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for QLearningBot")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor for QLearningBot")
    parser.add_argument("--no-pos", action="store_true", help="Exclude absolute position from state key (better generalization)")
    parser.add_argument("--potential", action="store_true", help="Enable potential-based shaping toward goal")
    parser.add_argument("--progress-scale", type=float, default=5.0, help="Scale for shaping progress toward goal")
    args = parser.parse_args()
    profileArg = str(args.profile)
    episodesArg = int(args.episodes)
    createArg = bool(args.create)
    saveJsonArg = bool(args.save_json)
    botArg = str(args.bot)
    lrArg = float(args.lr)
    gammaArg = float(args.gamma)
    noPosArg = bool(args.no_pos)
    potentialArg = bool(args.potential)
    progressScaleArg = float(args.progressScale)

    if createArg:
        createProfile(
            profileArg,
            botArg,
            lrArg,
            gammaArg,
            usePositionInState=(not noPosArg),
            potential=potentialArg,
            progressScale=progressScaleArg,
        )
        # If only creating, exit early unless episodes > 0
        if episodesArg <= 0:
            return

    if episodesArg <= 0:
        print("No episodes requested. Use --episodes N to run training.")
        return

    results = runProfileEpisodes(profileArg, episodesArg)

    summary = results["summary"]
    print("Profile:", summary["profile"]) 
    print("Episodes:", summary["episodes"]) 
    print("Success Rate:", f"{summary['success_rate']*100:.1f}%")
    print("Avg Reward:", f"{summary['avg_reward']:.2f}")
    print("Avg Walls Hit:", f"{summary['avg_walls_hit']:.2f}")
    print("Q-table Growth:", summary["qtable_growth"]) 
    print("Optimal Path Length:", summary["optimal_path_len"]) 

    # Brief per-episode line
    print("\nEpisodes:")
    for e in results["per_episode"]:
        print(
            f"#{e['episode']:03d}  R={e['reward']:.1f}  S={'Y' if e['success'] else 'N'}  "
            f"steps={e['steps_total']}  walls={e['walls_hit']}  q={e['q_table_states']} "
            f"uniq/opt={e['unique_vs_optimal']:.2f}"
        )

    if results["hints"]:
        print("\nHints:")
        for h in results["hints"]:
            print("-", h)

    if saveJsonArg:
        outDir = os.path.join("profiles", summary["profile"])
        os.makedirs(outDir, exist_ok=True)
        outPath = os.path.join(outDir, "debug_results.json")
        with open(outPath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {outPath}")


if __name__ == "__main__":
    main()
