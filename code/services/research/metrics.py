from __future__ import annotations

import datetime
from typing import Any

from services.episodeResult import EpisodeResult
from services.research.plan import ProfileSpec
from services.research.schema import normalize_export_keys

ROLLING_WINDOW: int = 50


def _mean(vals: list[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return round((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5, 4)


def _observed_failure_metric(step_limit_rate: float, no_progress_rate: float, wall_contact_rate: float) -> float:
    return round(float(step_limit_rate) + float(no_progress_rate) + float(wall_contact_rate), 4)


def episode_to_row(
    result: EpisodeResult,
    spec: ProfileSpec,
    episode_num: int,
    global_episode: int,
    phase: str,
    bot: Any,
    elapsed_seconds: float,
    experiment_name: str,
    maze_id: str,
) -> dict[str, Any]:
    cond = spec.condition
    path_inefficiency_ratio = (
        round(result.steps / result.optimal_steps, 4)
        if result.optimal_steps and result.optimal_steps > 0
        else None
    )
    decisions_raw = int(result.decisions)
    decisions = decisions_raw if decisions_raw > 0 else int(result.steps)
    option_selections = int(result.option_selections)
    option_steps = int(result.option_steps)
    mean_action_duration = round(float(result.steps) / float(decisions_raw), 4) if decisions_raw > 0 else None
    option_step_fraction = round(float(option_steps) / float(result.steps), 4) if int(result.steps) > 0 else 0.0
    step_limit_failure = int(result.outcome == "step_limit")
    no_progress_failure = int(result.outcome == "no_progress")
    wall_contact_failure = int(int(result.times_hit_wall) > 0)

    # QL-specific fields
    q_table_states = None
    ql_epsilon = None
    qLearner = getattr(bot, "qLearning", None)
    if qLearner is not None:
        q_table_states = len(qLearner.qTable)
        ql_epsilon = round(qLearner.explorationRate(), 6)

    # DQN-specific fields
    replay_size = None
    warmup_required = None
    warmup_complete = None
    training_updates = None
    dqn_epsilon = None
    dqn_loss = None
    epsilon_is_manual = False

    agent_obj = getattr(bot, "agent", None)
    if agent_obj is not None:
        # Use diagnostics() if available for a clean snapshot
        diag_fn = getattr(agent_obj, "diagnostics", None)
        if callable(diag_fn):
            diag = diag_fn()
            replay_size = getattr(diag, "replaySize", None)
            dqn_epsilon = round(float(diag.epsilon), 6) if diag.epsilon is not None else None
            training_updates = getattr(diag, "trainingUpdates", None)
            raw_loss = getattr(diag, "lastLoss", None)
            dqn_loss = round(float(raw_loss), 6) if raw_loss is not None else None
        else:
            replay = getattr(agent_obj, "replay", None)
            if replay is not None:
                replay_size = len(replay)
            eps_fn = getattr(agent_obj, "_epsilon", None)
            if callable(eps_fn):
                dqn_epsilon = round(float(eps_fn()), 6)
            training_updates = getattr(agent_obj, "trainingUpdates", None)
            raw_loss = getattr(agent_obj, "lastLoss", None)
            dqn_loss = round(float(raw_loss), 6) if raw_loss is not None else None

        config = getattr(bot, "config", None)
        warmup_required = getattr(config, "replayWarmupSteps", None)
        warmup_complete = not getattr(bot, "isWarmingUp", False)

    epsilon = ql_epsilon if ql_epsilon is not None else dqn_epsilon
    eval_epsilon = result.eval_epsilon if phase == "evaluation" else None
    if phase == "evaluation" and epsilon is not None:
        epsilon = 0.0
    if phase == "evaluation" and epsilon is None:
        epsilon = eval_epsilon

    return {
        "experiment_name": experiment_name,
        "profile_name": spec.profile_name,
        "agent_type": spec.agent_type,
        "bot_type": spec.bot_type,
        "condition": cond.condition_id,
        "maze_mode": cond.mode,
        "maze_size": f"{cond.width}x{cond.height}",
        "maze_id": maze_id,
        "run_id": spec.seed_label,
        "seed": spec.seed,
        "phase": phase,
        "episode": episode_num,
        "global_episode": global_episode,
        "success": int(result.success),
        "outcome": result.outcome,
        "total_reward": round(result.total_reward, 4),
        "steps": result.steps,
        "cumulative_env_steps": None,
        "cumulative_decisions": None,
        "decisions": decisions,
        "mean_action_duration": mean_action_duration,
        "option_selections": option_selections,
        "option_steps": option_steps,
        "option_step_fraction": option_step_fraction,
        "optimal_steps": result.optimal_steps,
        "path_inefficiency_ratio": path_inefficiency_ratio,
        "times_hit_wall": result.times_hit_wall,
        "step_limit_failure": step_limit_failure,
        "no_progress_failure": no_progress_failure,
        "wall_contact_failure": wall_contact_failure,
        "times_hit_enemy": 0,
        "enemy_kills": getattr(bot, "lastEpisodeEnemyKills", 0) or 0,
        "q_table_states": q_table_states,
        "replay_size": replay_size,
        "warmup_required": warmup_required,
        "warmup_complete": warmup_complete,
        "training_updates": training_updates,
        "epsilon": epsilon,
        "eval_epsilon": eval_epsilon,
        "epsilon_is_manual": int(epsilon_is_manual),
        "loss": dqn_loss,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def coerce_episode_row(row: dict[str, str]) -> dict[str, Any]:
    """Re-type a CSV row read back from disk to its original Python types."""
    normalized = dict(normalize_export_keys(row))
    if "path_inefficiency_ratio" not in normalized and "steps_over_optimal" in normalized:
        normalized["path_inefficiency_ratio"] = normalized.get("steps_over_optimal")
    normalized.pop("steps_over_optimal", None)
    _INT = {
        "seed", "episode", "global_episode", "success", "steps",
        "cumulative_env_steps", "cumulative_decisions", "optimal_steps",
        "decisions", "option_selections", "option_steps", "times_hit_wall",
        "step_limit_failure", "no_progress_failure", "wall_contact_failure",
        "times_hit_enemy", "enemy_kills", "q_table_states", "replay_size",
        "warmup_required", "training_updates", "epsilon_is_manual",
    }
    _FLOAT = {
        "total_reward", "path_inefficiency_ratio", "steps_over_optimal", "mean_action_duration",
        "option_step_fraction", "epsilon", "eval_epsilon", "loss", "elapsed_seconds",
    }
    _BOOL = {"warmup_complete"}
    out: dict[str, Any] = {}
    for key, value in normalized.items():
        if value == "":
            out[key] = None
        elif key in _INT:
            out[key] = int(value)
        elif key in _FLOAT:
            out[key] = float(value)
        elif key in _BOOL:
            out[key] = value.lower() == "true"
        else:
            out[key] = value
    return out


EPISODES_CSV_COLUMNS = [
    "experiment_name", "profile_name", "agent_type", "bot_type",
    "condition", "maze_mode", "maze_size", "maze_id", "run_id", "seed",
    "phase", "episode", "global_episode", "success", "outcome",
    "total_reward", "steps", "cumulative_env_steps", "cumulative_decisions",
    "optimal_steps", "path_inefficiency_ratio",
    "decisions", "mean_action_duration", "option_selections", "option_steps",
    "option_step_fraction", "times_hit_wall", "step_limit_failure",
    "no_progress_failure", "wall_contact_failure", "times_hit_enemy", "enemy_kills",
    "q_table_states", "replay_size", "warmup_required", "warmup_complete",
    "training_updates", "epsilon", "eval_epsilon", "epsilon_is_manual", "loss",
    "elapsed_seconds", "timestamp",
]


def compute_profile_summary(
    spec: ProfileSpec,
    training_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    bot: Any,
) -> dict[str, Any]:
    tr = training_rows
    ev = eval_rows

    t_success = [r["success"] for r in tr]
    t_rewards = [r["total_reward"] for r in tr]
    t_steps = [r["steps"] for r in tr]
    t_decisions = [r["decisions"] for r in tr]
    t_action_duration = [r["mean_action_duration"] for r in tr if r["mean_action_duration"] is not None]
    t_option_selections = [r["option_selections"] for r in tr]
    t_option_fraction = [r["option_step_fraction"] for r in tr]
    t_pir = [r["path_inefficiency_ratio"] for r in tr if r["path_inefficiency_ratio"] is not None]
    t_walls = [r["times_hit_wall"] for r in tr]
    t_step_limit = [r["step_limit_failure"] for r in tr]
    t_no_progress = [r["no_progress_failure"] for r in tr]
    t_wall_contact = [r["wall_contact_failure"] for r in tr]

    first_success: int | None = None
    for r in tr:
        if r["success"]:
            first_success = r["episode"]
            break

    final_50 = tr[-ROLLING_WINDOW:] if len(tr) >= ROLLING_WINDOW else tr
    f50_success = [r["success"] for r in final_50]
    f50_rewards = [r["total_reward"] for r in final_50]

    e_success = [r["success"] for r in ev]
    e_rewards = [r["total_reward"] for r in ev]
    e_steps = [r["steps"] for r in ev]
    e_decisions = [r["decisions"] for r in ev]
    e_action_duration = [r["mean_action_duration"] for r in ev if r["mean_action_duration"] is not None]
    e_option_selections = [r["option_selections"] for r in ev]
    e_option_fraction = [r["option_step_fraction"] for r in ev]
    e_pir = [r["path_inefficiency_ratio"] for r in ev if r["path_inefficiency_ratio"] is not None]
    e_walls = [r["times_hit_wall"] for r in ev]
    e_step_limit = [r["step_limit_failure"] for r in ev]
    e_no_progress = [r["no_progress_failure"] for r in ev]
    e_wall_contact = [r["wall_contact_failure"] for r in ev]

    t_step_limit_rate = _mean([float(v) for v in t_step_limit])
    t_no_progress_rate = _mean([float(v) for v in t_no_progress])
    t_wall_contact_rate = _mean([float(v) for v in t_wall_contact])
    e_step_limit_rate = _mean([float(v) for v in e_step_limit])
    e_no_progress_rate = _mean([float(v) for v in e_no_progress])
    e_wall_contact_rate = _mean([float(v) for v in e_wall_contact])

    # DQN diagnostics
    dqn_dx: dict[str, Any] = {"final_replay_size": None, "warmup_required": None,
                               "warmup_complete": None, "final_training_updates": None,
                               "final_epsilon": None, "final_loss": None}
    ql_dx: dict[str, Any] = {"final_q_table_states": None, "final_epsilon": None}

    if tr:
        last = tr[-1]
        if last.get("replay_size") is not None:
            dqn_dx["final_replay_size"] = last["replay_size"]
            dqn_dx["warmup_required"] = last.get("warmup_required")
            dqn_dx["warmup_complete"] = last.get("warmup_complete")
            dqn_dx["final_training_updates"] = last.get("training_updates")
            dqn_dx["final_epsilon"] = last.get("epsilon")
            dqn_dx["final_loss"] = last.get("loss")
        if last.get("q_table_states") is not None:
            ql_dx["final_q_table_states"] = last["q_table_states"]
            ql_dx["final_epsilon"] = last.get("epsilon")

    option_diagnostics: dict[str, Any] = {}
    option_diag_fn = getattr(bot, "optionDiagnostics", None)
    if callable(option_diag_fn):
        option_diagnostics = dict(option_diag_fn())

    return {
        "profile_name": spec.profile_name,
        "agent_type": spec.agent_type,
        "bot_type": spec.bot_type,
        "condition": spec.condition.condition_id,
        "run_id": spec.seed_label,
        "seed": spec.seed,
        "training_episodes": len(tr),
        "evaluation_episodes": len(ev),
        "training": {
            "success_rate": _mean([float(s) for s in t_success]),
            "mean_reward": _mean(t_rewards),
            "std_reward": _std(t_rewards),
            "mean_steps": _mean([float(s) for s in t_steps]),
            "mean_decisions": _mean([float(s) for s in t_decisions]),
            "mean_action_duration": _mean([float(v) for v in t_action_duration]),
            "mean_option_selections": _mean([float(v) for v in t_option_selections]),
            "mean_option_step_fraction": _mean([float(v) for v in t_option_fraction]),
            "mean_path_inefficiency_ratio": _mean(t_pir),
            "mean_wall_hits": _mean([float(w) for w in t_walls]),
            "step_limit_rate": t_step_limit_rate,
            "no_progress_rate": t_no_progress_rate,
            "wall_contact_rate": t_wall_contact_rate,
            "R_m_observed": _observed_failure_metric(
                t_step_limit_rate,
                t_no_progress_rate,
                t_wall_contact_rate,
            ),
            "first_success_episode": first_success,
            "final_50_success_rate": _mean([float(s) for s in f50_success]),
            "final_50_mean_reward": _mean(f50_rewards),
        },
        "evaluation": {
            "success_rate": _mean([float(s) for s in e_success]),
            "mean_reward": _mean(e_rewards),
            "std_reward": _std(e_rewards),
            "mean_steps": _mean([float(s) for s in e_steps]),
            "std_steps": _std([float(s) for s in e_steps]),
            "mean_decisions": _mean([float(s) for s in e_decisions]),
            "mean_action_duration": _mean([float(v) for v in e_action_duration]),
            "mean_option_selections": _mean([float(v) for v in e_option_selections]),
            "mean_option_step_fraction": _mean([float(v) for v in e_option_fraction]),
            "mean_path_inefficiency_ratio": _mean(e_pir),
            "mean_wall_hits": _mean([float(w) for w in e_walls]),
            "step_limit_rate": e_step_limit_rate,
            "no_progress_rate": e_no_progress_rate,
            "wall_contact_rate": e_wall_contact_rate,
            "R_m_observed": _observed_failure_metric(
                e_step_limit_rate,
                e_no_progress_rate,
                e_wall_contact_rate,
            ),
        },
        "dqn_diagnostics": dqn_dx,
        "qlearning_diagnostics": ql_dx,
        "option_diagnostics": option_diagnostics,
    }
