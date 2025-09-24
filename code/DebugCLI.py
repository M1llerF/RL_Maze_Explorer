import argparse
import json
import os
import pickle
from typing import Any, Dict, List
from services.repository import ArtifactsRepository

from GameEnvironment import GameEnvironment
from BotConfigs import QLearningConfig, bot_configs
from RewardSystem import RewardConfig


def _read_profile_stats(profile_name: str) -> Dict[str, Any]:
    """Read aggregated profile stats via repository, returning defaults when absent."""
    try:
        repo = ArtifactsRepository("profiles")
        return repo.read_profile_stats(profile_name) or {}
    except Exception:
        return {}


def _delta(after: Dict[str, Any], before: Dict[str, Any], key: str) -> int:
    return int(after.get(key, 0) - before.get(key, 0))


def run_profile_episodes(profile_name: str, episodes: int) -> Dict[str, Any]:
    """
    Run a profile for N episodes without GUI and collect debug metrics.

    Returns a dictionary with per-episode metrics and summary stats.
    """
    env = GameEnvironment()

    # Load profile and create/apply bot
    profile = env.profile_manager.load_profile(profile_name)
    bot_index = env.apply_profile(profile)
    bot = env.bots[bot_index]

    # Static reference values
    optimal_path = bot.tools.get_optimal_path_info(bot.maze.start, bot.maze.end, output="path")
    optimal_len = len(optimal_path)

    per_episode: List[Dict[str, Any]] = []

    for ep in range(episodes):
        before = _read_profile_stats(profile_name)

        bot.run_episode()

        after = _read_profile_stats(profile_name)

        success = bot.position == bot.maze.end

        steps_total = _delta(after, before, "total_steps")
        steps_unique = _delta(after, before, "non_repeating_steps_taken")
        steps_revisited = _delta(after, before, "times_revisited_squares")
        walls_hit = _delta(after, before, "times_hit_wall")

        per_episode.append({
            "episode": ep + 1,
            "reward": float(bot.total_reward),
            "success": bool(success),
            "steps_total": int(steps_total),
            "steps_unique": int(steps_unique),
            "steps_revisited": int(steps_revisited),
            "walls_hit": int(walls_hit),
            "q_table_states": int(len(bot.q_learning.q_table)),
            "unique_vs_optimal": float(steps_unique / optimal_len) if optimal_len > 0 else 0.0,
        })

        # Reset environment for next episode
        env.reset_environment(bot_index)

    # Summaries
    rewards = [e["reward"] for e in per_episode]
    successes = [1 if e["success"] else 0 for e in per_episode]
    q_sizes = [e["q_table_states"] for e in per_episode]
    walls = [e["walls_hit"] for e in per_episode]

    summary = {
        "episodes": episodes,
        "profile": profile_name,
        "success_rate": sum(successes) / episodes if episodes else 0.0,
        "avg_reward": sum(rewards) / episodes if episodes else 0.0,
        "avg_walls_hit": sum(walls) / episodes if episodes else 0.0,
        "qtable_growth": q_sizes[-1] - q_sizes[0] if episodes > 1 else 0,
        "optimal_path_len": optimal_len,
    }

    # Simple heuristics for failure hints
    hints: List[str] = []
    if summary["success_rate"] < 0.25 and summary["avg_reward"] < 0:
        hints.append("Low success and negative rewards: agent likely not reaching goal; consider lowering exploration decay or adjusting rewards.")
    if summary["avg_walls_hit"] > 5:
        hints.append("High wall collisions: increase wall penalty or improve state features.")
    if summary["qtable_growth"] <= 0 and episodes > 5:
        hints.append("Q-table not growing: exploration may be too low or episodes too short.")

    return {"summary": summary, "per_episode": per_episode, "hints": hints}


def create_profile(profile_name: str, bot_type: str = "QLearningBot", lr: float = 0.1, gamma: float = 0.9,
                   use_position_in_state: bool = True, potential: bool = False, progress_scale: float = 5.0) -> None:
    """Create a new profile directory with default config and reward settings."""
    env = GameEnvironment()

    if bot_type != "QLearningBot":
        raise ValueError("Currently only QLearningBot is supported by this CLI.")

    # Build config
    config = QLearningConfig(learning_rate=lr, discount_factor=gamma, use_position_in_state=use_position_in_state)

    # Build reward config from bot_configs defaults
    rewards = bot_configs.get(bot_type, {}).get("rewards", {})
    reward_config = RewardConfig(reward_modifiers=rewards, use_potential_shaping=potential, progress_scale=progress_scale)

    env.setup_new_profile(profile_name, bot_type, config, reward_config)
    print(f"Created profile '{profile_name}' for {bot_type} (lr={lr}, gamma={gamma}, pos_state={use_position_in_state}, potential={potential}, progress_scale={progress_scale}).")


def main():
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

    if args.create:
        create_profile(
            args.profile,
            args.bot,
            args.lr,
            args.gamma,
            use_position_in_state=(not args.no_pos),
            potential=args.potential,
            progress_scale=args.progress_scale,
        )
        # If only creating, exit early unless episodes > 0
        if args.episodes <= 0:
            return

    if args.episodes <= 0:
        print("No episodes requested. Use --episodes N to run training.")
        return

    results = run_profile_episodes(args.profile, args.episodes)

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

    if args.save_json:
        out_dir = os.path.join("profiles", summary["profile"])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "debug_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
