from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any

from services.research.metrics import EPISODES_CSV_COLUMNS, ROLLING_WINDOW, _mean, _std, coerce_episode_row
from services.research.plan import ConditionSpec, _parse_condition as _condition_from_raw
from services.research.profileBuilder import SHARED_REWARD_MODIFIERS, build_config_for_agent
from services.research.schema import camelize_export_keys, normalize_export_keys, snake_to_camel


DEFAULT_LAMBDA_FAILURE: float = 1.0
LOOPS_NOT_DIRECTLY_LOGGED_NOTE: str = "Loops are not directly logged in the current experiment."


def write_json(path: str, data: Any, *, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(camelize_export_keys(data), f, indent=indent, default=str)


def write_csv(path: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    export_columns = [snake_to_camel(column) for column in columns]
    export_rows = [{snake_to_camel(key): value for key, value in row.items()} for row in rows]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=export_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(export_rows)


def append_csv_rows(path: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    export_columns = [snake_to_camel(column) for column in columns]
    export_rows = [{snake_to_camel(key): value for key, value in row.items()} for row in rows]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=export_columns, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerows(export_rows)


def write_episodes_csv(path: str, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, EPISODES_CSV_COLUMNS)


def write_profile_training_csv(profile_out_dir: str, rows: list[dict[str, Any]]) -> None:
    write_csv(os.path.join(profile_out_dir, "training_episodes.csv"), rows, EPISODES_CSV_COLUMNS)


def write_profile_eval_csv(profile_out_dir: str, rows: list[dict[str, Any]]) -> None:
    write_csv(os.path.join(profile_out_dir, "evaluation_episodes.csv"), rows, EPISODES_CSV_COLUMNS)


def write_profile_summary(profile_out_dir: str, summary: dict[str, Any]) -> None:
    write_json(os.path.join(profile_out_dir, "summary.json"), summary)


def write_profile_diagnostics(profile_out_dir: str, data: dict[str, Any]) -> None:
    write_json(os.path.join(profile_out_dir, "diagnostics.json"), data)


def write_heatmap(profile_out_dir: str, phase: str, heatmap_data: dict[str, Any]) -> None:
    heatmaps_dir = os.path.join(profile_out_dir, "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)
    write_json(os.path.join(heatmaps_dir, f"{phase}_heatmap.json"), heatmap_data)


def write_summary_by_profile(out_dir: str, summaries: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        training = summary.get("training", {})
        evaluation = summary.get("evaluation", {})
        option_diag = summary.get("option_diagnostics", {})
        rows.append(
            {
                "profile_name": summary["profile_name"],
                "agent_type": summary["agent_type"],
                "bot_type": summary["bot_type"],
                "condition": summary["condition"],
                "run_id": summary["run_id"],
                "seed": summary["seed"],
                "training_episodes": summary["training_episodes"],
                "evaluation_episodes": summary["evaluation_episodes"],
                "train_success_rate": training.get("success_rate"),
                "train_mean_reward": training.get("mean_reward"),
                "train_std_reward": training.get("std_reward"),
                "train_mean_steps": training.get("mean_steps"),
                "train_mean_decisions": training.get("mean_decisions"),
                "train_mean_action_duration": training.get("mean_action_duration"),
                "train_mean_option_selections": training.get("mean_option_selections"),
                "train_mean_option_step_fraction": training.get("mean_option_step_fraction"),
                "train_step_limit_rate": training.get("step_limit_rate"),
                "train_no_progress_rate": training.get("no_progress_rate"),
                "train_wall_contact_rate": training.get("wall_contact_rate"),
                "train_R_m_observed": training.get("R_m_observed"),
                "train_first_success": training.get("first_success_episode"),
                "train_final_50_success_rate": training.get("final_50_success_rate"),
                "train_final_50_mean_reward": training.get("final_50_mean_reward"),
                "eval_success_rate": evaluation.get("success_rate"),
                "eval_mean_reward": evaluation.get("mean_reward"),
                "eval_std_reward": evaluation.get("std_reward"),
                "eval_mean_steps": evaluation.get("mean_steps"),
                "eval_std_steps": evaluation.get("std_steps"),
                "eval_mean_decisions": evaluation.get("mean_decisions"),
                "eval_mean_action_duration": evaluation.get("mean_action_duration"),
                "eval_mean_option_selections": evaluation.get("mean_option_selections"),
                "eval_mean_option_step_fraction": evaluation.get("mean_option_step_fraction"),
                "eval_step_limit_rate": evaluation.get("step_limit_rate"),
                "eval_no_progress_rate": evaluation.get("no_progress_rate"),
                "eval_wall_contact_rate": evaluation.get("wall_contact_rate"),
                "eval_R_m_observed": evaluation.get("R_m_observed"),
                "eval_mean_path_inefficiency_ratio": evaluation.get("mean_path_inefficiency_ratio"),
                "eval_mean_wall_hits": evaluation.get("mean_wall_hits"),
                "macro_option_set": option_diag.get("macro_option_set"),
                "total_option_selections": option_diag.get("total_option_selections"),
            }
        )
    columns = list(rows[0].keys()) if rows else []
    write_csv(os.path.join(out_dir, "summary_by_profile.csv"), rows, columns)


def write_summary_by_agent_condition(out_dir: str, summaries: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        groups[(str(summary["agent_type"]), str(summary["condition"]))].append(summary)

    rows: list[dict[str, Any]] = []
    for (agent, condition), group in sorted(groups.items()):
        train_success = [float(s.get("training", {}).get("success_rate", 0.0)) for s in group]
        train_reward = [float(s.get("training", {}).get("mean_reward", 0.0)) for s in group]
        train_steps = [float(s.get("training", {}).get("mean_steps", 0.0)) for s in group]
        train_decisions = [float(s.get("training", {}).get("mean_decisions", 0.0)) for s in group]
        train_action_duration = [float(s.get("training", {}).get("mean_action_duration", 0.0)) for s in group]
        train_option_fraction = [float(s.get("training", {}).get("mean_option_step_fraction", 0.0)) for s in group]
        train_step_limit = [float(s.get("training", {}).get("step_limit_rate", 0.0)) for s in group]
        train_no_progress = [float(s.get("training", {}).get("no_progress_rate", 0.0)) for s in group]
        train_wall_contact = [float(s.get("training", {}).get("wall_contact_rate", 0.0)) for s in group]
        train_r_m_observed = [float(s.get("training", {}).get("R_m_observed", 0.0)) for s in group]
        train_first_success = [
            float(v)
            for s in group
            for v in [s.get("training", {}).get("first_success_episode")]
            if v is not None
        ]
        train_final_50_success = [
            float(s.get("training", {}).get("final_50_success_rate", 0.0)) for s in group
        ]
        eval_success = [float(s.get("evaluation", {}).get("success_rate", 0.0)) for s in group]
        eval_reward = [float(s.get("evaluation", {}).get("mean_reward", 0.0)) for s in group]
        eval_steps = [float(s.get("evaluation", {}).get("mean_steps", 0.0)) for s in group]
        eval_decisions = [float(s.get("evaluation", {}).get("mean_decisions", 0.0)) for s in group]
        eval_action_duration = [float(s.get("evaluation", {}).get("mean_action_duration", 0.0)) for s in group]
        eval_option_fraction = [float(s.get("evaluation", {}).get("mean_option_step_fraction", 0.0)) for s in group]
        eval_step_limit = [float(s.get("evaluation", {}).get("step_limit_rate", 0.0)) for s in group]
        eval_no_progress = [float(s.get("evaluation", {}).get("no_progress_rate", 0.0)) for s in group]
        eval_wall_contact = [float(s.get("evaluation", {}).get("wall_contact_rate", 0.0)) for s in group]
        eval_r_m_observed = [float(s.get("evaluation", {}).get("R_m_observed", 0.0)) for s in group]
        eval_path_inefficiency = [
            float(s.get("evaluation", {}).get("mean_path_inefficiency_ratio", 0.0)) for s in group
        ]
        eval_walls = [float(s.get("evaluation", {}).get("mean_wall_hits", 0.0)) for s in group]

        rows.append(
            {
                "agent_type": agent,
                "condition": condition,
                "n_profiles": len(group),
                "train_success_rate_mean": _mean(train_success),
                "train_success_rate_std": _std(train_success),
                "train_mean_reward_mean": _mean(train_reward),
                "train_mean_reward_std": _std(train_reward),
                "train_mean_steps_mean": _mean(train_steps),
                "train_mean_steps_std": _std(train_steps),
                "train_mean_decisions_mean": _mean(train_decisions),
                "train_mean_decisions_std": _std(train_decisions),
                "train_mean_action_duration_mean": _mean(train_action_duration),
                "train_mean_option_step_fraction_mean": _mean(train_option_fraction),
                "train_step_limit_rate_mean": _mean(train_step_limit),
                "train_step_limit_rate_std": _std(train_step_limit),
                "train_no_progress_rate_mean": _mean(train_no_progress),
                "train_no_progress_rate_std": _std(train_no_progress),
                "train_wall_contact_rate_mean": _mean(train_wall_contact),
                "train_wall_contact_rate_std": _std(train_wall_contact),
                "train_R_m_observed_mean": _mean(train_r_m_observed),
                "train_R_m_observed_std": _std(train_r_m_observed),
                "train_first_success_episode_mean": _mean(train_first_success),
                "train_first_success_episode_std": _std(train_first_success),
                "train_final_50_success_rate_mean": _mean(train_final_50_success),
                "train_final_50_success_rate_std": _std(train_final_50_success),
                "eval_success_rate_mean": _mean(eval_success),
                "eval_success_rate_std": _std(eval_success),
                "eval_mean_reward_mean": _mean(eval_reward),
                "eval_mean_reward_std": _std(eval_reward),
                "eval_mean_steps_mean": _mean(eval_steps),
                "eval_mean_steps_std": _std(eval_steps),
                "eval_mean_decisions_mean": _mean(eval_decisions),
                "eval_mean_decisions_std": _std(eval_decisions),
                "eval_mean_action_duration_mean": _mean(eval_action_duration),
                "eval_mean_option_step_fraction_mean": _mean(eval_option_fraction),
                "eval_step_limit_rate_mean": _mean(eval_step_limit),
                "eval_step_limit_rate_std": _std(eval_step_limit),
                "eval_no_progress_rate_mean": _mean(eval_no_progress),
                "eval_no_progress_rate_std": _std(eval_no_progress),
                "eval_wall_contact_rate_mean": _mean(eval_wall_contact),
                "eval_wall_contact_rate_std": _std(eval_wall_contact),
                "eval_R_m_observed_mean": _mean(eval_r_m_observed),
                "eval_R_m_observed_std": _std(eval_r_m_observed),
                "eval_mean_path_inefficiency_ratio_mean": _mean(eval_path_inefficiency),
                "eval_mean_path_inefficiency_ratio_std": _std(eval_path_inefficiency),
                "eval_mean_wall_hits_mean": _mean(eval_walls),
                "eval_mean_wall_hits_std": _std(eval_walls),
            }
        )

    columns = list(rows[0].keys()) if rows else []
    write_csv(os.path.join(out_dir, "summary_by_agent_condition.csv"), rows, columns)


def write_plot_data(out_dir: str, all_rows: list[dict[str, Any]]) -> None:
    plot_dir = os.path.join(out_dir, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)

    window = ROLLING_WINDOW

    def rolling_mean(vals: list[float], width: int) -> list[float]:
        result: list[float] = []
        for i in range(len(vals)):
            sample = vals[max(0, i - width + 1) : i + 1]
            result.append(round(sum(sample) / len(sample), 4))
        return result

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        key = (str(row["agent_type"]), str(row["condition"]), str(row["run_id"]), str(row["phase"]))
        groups[key].append(row)

    reward_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    path_inefficiency_rows: list[dict[str, Any]] = []
    wallhit_rows: list[dict[str, Any]] = []

    for (agent, condition, run_id, phase), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["episode"])
        rewards = [float(row["total_reward"]) for row in rows]
        successes = [float(row["success"]) for row in rows]
        path_inefficiency = [
            float(row["path_inefficiency_ratio"]) if row.get("path_inefficiency_ratio") is not None else 0.0
            for row in rows
        ]
        walls = [float(row["times_hit_wall"]) for row in rows]

        reward_roll = rolling_mean(rewards, window)
        success_roll = rolling_mean(successes, window)
        path_inefficiency_roll = rolling_mean(path_inefficiency, window)
        wall_roll = rolling_mean(walls, window)

        for i, row in enumerate(rows):
            base = {
                "agent_type": agent,
                "condition": condition,
                "run_id": run_id,
                "phase": phase,
                "episode": row["episode"],
                "cumulative_env_steps": row.get("cumulative_env_steps"),
                "cumulative_decisions": row.get("cumulative_decisions"),
            }
            reward_rows.append({**base, "value": rewards[i], "rolling_mean": reward_roll[i]})
            path_inefficiency_rows.append(
                {**base, "value": path_inefficiency[i], "rolling_mean": path_inefficiency_roll[i]}
            )
            wallhit_rows.append({**base, "value": walls[i], "rolling_mean": wall_roll[i]})

        for block_start in range(0, len(rows), window):
            block = rows[block_start : block_start + window]
            block_success = sum(row["success"] for row in block) / len(block)
            success_rows.append(
                {
                    "agent_type": agent,
                    "condition": condition,
                    "run_id": run_id,
                    "phase": phase,
                    "episode_start": block[0]["episode"],
                    "episode_end": block[-1]["episode"],
                    "success_rate": round(block_success, 4),
                }
            )

    curve_cols = [
        "agent_type", "condition", "run_id", "episode", "phase",
        "cumulative_env_steps", "cumulative_decisions", "value", "rolling_mean",
    ]
    success_cols = ["agent_type", "condition", "run_id", "phase", "episode_start", "episode_end", "success_rate"]

    write_csv(os.path.join(plot_dir, "reward_curve.csv"), reward_rows, curve_cols)
    write_csv(os.path.join(plot_dir, "success_curve.csv"), success_rows, success_cols)
    write_csv(os.path.join(plot_dir, "path_inefficiency_curve.csv"), path_inefficiency_rows, curve_cols)
    write_csv(os.path.join(plot_dir, "wall_hits_curve.csv"), wallhit_rows, curve_cols)

    eval_rows = [row for row in all_rows if row["phase"] == "evaluation"]
    eval_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        eval_groups[(str(row["agent_type"]), str(row["condition"]))].append(row)

    bar_rows: list[dict[str, Any]] = []
    for (agent, condition), rows in sorted(eval_groups.items()):
        success_vals = [float(row["success"]) for row in rows]
        reward_vals = [float(row["total_reward"]) for row in rows]
        n = len(success_vals)
        bar_rows.append(
            {
                "agent_type": agent,
                "condition": condition,
                "mean_success_rate": round(sum(success_vals) / n, 4) if n else 0.0,
                "mean_reward": round(sum(reward_vals) / n, 4) if n else 0.0,
                "n_episodes": n,
            }
        )
    write_csv(
        os.path.join(plot_dir, "final_eval_bars.csv"),
        bar_rows,
        ["agent_type", "condition", "mean_success_rate", "mean_reward", "n_episodes"],
    )


def _coerce_summary_by_profile_row(row: dict[str, str]) -> dict[str, Any]:
    normalized = dict(normalize_export_keys(row))
    if "eval_mean_path_inefficiency_ratio" not in normalized and "eval_mean_steps_over_optimal" in normalized:
        normalized["eval_mean_path_inefficiency_ratio"] = normalized.get("eval_mean_steps_over_optimal")
    normalized.pop("eval_mean_steps_over_optimal", None)
    int_fields = {
        "seed", "training_episodes", "evaluation_episodes", "train_first_success",
        "total_option_selections",
    }
    float_fields = {
        "train_success_rate", "train_mean_reward", "train_std_reward", "train_mean_steps",
        "train_mean_decisions", "train_mean_action_duration", "train_mean_option_selections",
        "train_mean_option_step_fraction", "train_step_limit_rate", "train_no_progress_rate",
        "train_wall_contact_rate", "train_R_m_observed", "train_final_50_success_rate",
        "train_final_50_mean_reward", "eval_success_rate", "eval_mean_reward",
        "eval_std_reward", "eval_mean_steps", "eval_std_steps", "eval_mean_decisions",
        "eval_mean_action_duration", "eval_mean_option_selections",
        "eval_mean_option_step_fraction", "eval_step_limit_rate", "eval_no_progress_rate",
        "eval_wall_contact_rate", "eval_R_m_observed", "eval_mean_path_inefficiency_ratio",
        "eval_mean_steps_over_optimal",
        "eval_mean_wall_hits",
    }
    out: dict[str, Any] = {}
    for key, value in normalized.items():
        if value == "":
            out[key] = None
        elif key in int_fields:
            out[key] = int(value)
        elif key in float_fields:
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _coerce_summary_by_agent_condition_row(row: dict[str, str]) -> dict[str, Any]:
    normalized = dict(normalize_export_keys(row))
    if (
        "eval_mean_path_inefficiency_ratio_mean" not in normalized
        and "eval_mean_steps_over_optimal_mean" in normalized
    ):
        normalized["eval_mean_path_inefficiency_ratio_mean"] = normalized.get("eval_mean_steps_over_optimal_mean")
    if (
        "eval_mean_path_inefficiency_ratio_std" not in normalized
        and "eval_mean_steps_over_optimal_std" in normalized
    ):
        normalized["eval_mean_path_inefficiency_ratio_std"] = normalized.get("eval_mean_steps_over_optimal_std")
    normalized.pop("eval_mean_steps_over_optimal_mean", None)
    normalized.pop("eval_mean_steps_over_optimal_std", None)
    int_fields = {"n_profiles"}
    float_fields = {
        "train_success_rate_mean", "train_success_rate_std",
        "train_mean_reward_mean", "train_mean_reward_std",
        "train_mean_steps_mean", "train_mean_steps_std",
        "train_mean_decisions_mean", "train_mean_decisions_std",
        "train_mean_action_duration_mean", "train_mean_option_step_fraction_mean",
        "train_step_limit_rate_mean", "train_step_limit_rate_std",
        "train_no_progress_rate_mean", "train_no_progress_rate_std",
        "train_wall_contact_rate_mean", "train_wall_contact_rate_std",
        "train_R_m_observed_mean", "train_R_m_observed_std",
        "train_first_success_episode_mean", "train_first_success_episode_std",
        "train_final_50_success_rate_mean", "train_final_50_success_rate_std",
        "eval_success_rate_mean", "eval_success_rate_std",
        "eval_mean_reward_mean", "eval_mean_reward_std",
        "eval_mean_steps_mean", "eval_mean_steps_std",
        "eval_mean_decisions_mean", "eval_mean_decisions_std",
        "eval_mean_action_duration_mean", "eval_mean_option_step_fraction_mean",
        "eval_step_limit_rate_mean", "eval_step_limit_rate_std",
        "eval_no_progress_rate_mean", "eval_no_progress_rate_std",
        "eval_wall_contact_rate_mean", "eval_wall_contact_rate_std",
        "eval_R_m_observed_mean", "eval_R_m_observed_std",
        "eval_mean_path_inefficiency_ratio_mean", "eval_mean_path_inefficiency_ratio_std",
        "eval_mean_steps_over_optimal_mean", "eval_mean_steps_over_optimal_std",
        "eval_mean_wall_hits_mean", "eval_mean_wall_hits_std",
    }
    out: dict[str, Any] = {}
    for key, value in normalized.items():
        if value == "":
            out[key] = None
        elif key in int_fields:
            out[key] = int(value)
        elif key in float_fields:
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _observed_failure_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    step_limit_rate = _mean([float(int(str(row.get("outcome", "")) == "step_limit")) for row in rows])
    no_progress_rate = _mean([float(int(str(row.get("outcome", "")) == "no_progress")) for row in rows])
    wall_contact_rate = _mean([float(int((row.get("times_hit_wall") or 0) > 0)) for row in rows])
    return {
        "step_limit_rate": step_limit_rate,
        "no_progress_rate": no_progress_rate,
        "wall_contact_rate": wall_contact_rate,
        "R_m_observed": round(step_limit_rate + no_progress_rate + wall_contact_rate, 4),
    }


def _aggregate_summary_by_agent_condition_rows_from_profile_rows(
    profile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        groups[(str(row["agent_type"]), str(row["condition"]))].append(row)

    rows: list[dict[str, Any]] = []
    for (agent, condition), group in sorted(groups.items()):
        train_success = [float(r.get("train_success_rate", 0.0) or 0.0) for r in group]
        train_reward = [float(r.get("train_mean_reward", 0.0) or 0.0) for r in group]
        train_steps = [float(r.get("train_mean_steps", 0.0) or 0.0) for r in group]
        train_decisions = [float(r.get("train_mean_decisions", 0.0) or 0.0) for r in group]
        train_action_duration = [float(r.get("train_mean_action_duration", 0.0) or 0.0) for r in group]
        train_option_fraction = [float(r.get("train_mean_option_step_fraction", 0.0) or 0.0) for r in group]
        train_step_limit = [float(r.get("train_step_limit_rate", 0.0) or 0.0) for r in group]
        train_no_progress = [float(r.get("train_no_progress_rate", 0.0) or 0.0) for r in group]
        train_wall_contact = [float(r.get("train_wall_contact_rate", 0.0) or 0.0) for r in group]
        train_r_m_observed = [float(r.get("train_R_m_observed", 0.0) or 0.0) for r in group]
        train_first_success = [
            float(v) for r in group for v in [r.get("train_first_success")] if v is not None
        ]
        train_final_50_success = [
            float(r.get("train_final_50_success_rate", 0.0) or 0.0) for r in group
        ]
        eval_success = [float(r.get("eval_success_rate", 0.0) or 0.0) for r in group]
        eval_reward = [float(r.get("eval_mean_reward", 0.0) or 0.0) for r in group]
        eval_steps = [float(r.get("eval_mean_steps", 0.0) or 0.0) for r in group]
        eval_decisions = [float(r.get("eval_mean_decisions", 0.0) or 0.0) for r in group]
        eval_action_duration = [float(r.get("eval_mean_action_duration", 0.0) or 0.0) for r in group]
        eval_option_fraction = [float(r.get("eval_mean_option_step_fraction", 0.0) or 0.0) for r in group]
        eval_step_limit = [float(r.get("eval_step_limit_rate", 0.0) or 0.0) for r in group]
        eval_no_progress = [float(r.get("eval_no_progress_rate", 0.0) or 0.0) for r in group]
        eval_wall_contact = [float(r.get("eval_wall_contact_rate", 0.0) or 0.0) for r in group]
        eval_r_m_observed = [float(r.get("eval_R_m_observed", 0.0) or 0.0) for r in group]
        eval_path_inefficiency = [
            float(r.get("eval_mean_path_inefficiency_ratio", 0.0) or 0.0) for r in group
        ]
        eval_walls = [float(r.get("eval_mean_wall_hits", 0.0) or 0.0) for r in group]

        rows.append(
            {
                "agent_type": agent,
                "condition": condition,
                "n_profiles": len(group),
                "train_success_rate_mean": _mean(train_success),
                "train_success_rate_std": _std(train_success),
                "train_mean_reward_mean": _mean(train_reward),
                "train_mean_reward_std": _std(train_reward),
                "train_mean_steps_mean": _mean(train_steps),
                "train_mean_steps_std": _std(train_steps),
                "train_mean_decisions_mean": _mean(train_decisions),
                "train_mean_decisions_std": _std(train_decisions),
                "train_mean_action_duration_mean": _mean(train_action_duration),
                "train_mean_option_step_fraction_mean": _mean(train_option_fraction),
                "train_step_limit_rate_mean": _mean(train_step_limit),
                "train_step_limit_rate_std": _std(train_step_limit),
                "train_no_progress_rate_mean": _mean(train_no_progress),
                "train_no_progress_rate_std": _std(train_no_progress),
                "train_wall_contact_rate_mean": _mean(train_wall_contact),
                "train_wall_contact_rate_std": _std(train_wall_contact),
                "train_R_m_observed_mean": _mean(train_r_m_observed),
                "train_R_m_observed_std": _std(train_r_m_observed),
                "train_first_success_episode_mean": _mean(train_first_success),
                "train_first_success_episode_std": _std(train_first_success),
                "train_final_50_success_rate_mean": _mean(train_final_50_success),
                "train_final_50_success_rate_std": _std(train_final_50_success),
                "eval_success_rate_mean": _mean(eval_success),
                "eval_success_rate_std": _std(eval_success),
                "eval_mean_reward_mean": _mean(eval_reward),
                "eval_mean_reward_std": _std(eval_reward),
                "eval_mean_steps_mean": _mean(eval_steps),
                "eval_mean_steps_std": _std(eval_steps),
                "eval_mean_decisions_mean": _mean(eval_decisions),
                "eval_mean_decisions_std": _std(eval_decisions),
                "eval_mean_action_duration_mean": _mean(eval_action_duration),
                "eval_mean_option_step_fraction_mean": _mean(eval_option_fraction),
                "eval_step_limit_rate_mean": _mean(eval_step_limit),
                "eval_step_limit_rate_std": _std(eval_step_limit),
                "eval_no_progress_rate_mean": _mean(eval_no_progress),
                "eval_no_progress_rate_std": _std(eval_no_progress),
                "eval_wall_contact_rate_mean": _mean(eval_wall_contact),
                "eval_wall_contact_rate_std": _std(eval_wall_contact),
                "eval_R_m_observed_mean": _mean(eval_r_m_observed),
                "eval_R_m_observed_std": _std(eval_r_m_observed),
                "eval_mean_path_inefficiency_ratio_mean": _mean(eval_path_inefficiency),
                "eval_mean_path_inefficiency_ratio_std": _std(eval_path_inefficiency),
                "eval_mean_wall_hits_mean": _mean(eval_walls),
                "eval_mean_wall_hits_std": _std(eval_walls),
            }
        )
    return rows


def refresh_observed_failure_metrics_from_episodes(out_dir: str) -> None:
    episodes_path = os.path.join(out_dir, "episodes.csv")
    summary_by_profile_path = os.path.join(out_dir, "summary_by_profile.csv")
    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Missing episodes export: {episodes_path}")
    if not os.path.exists(summary_by_profile_path):
        raise FileNotFoundError(f"Missing summary export: {summary_by_profile_path}")

    episode_rows = [coerce_episode_row(row) for row in _read_csv_rows(episodes_path)]
    profile_rows = [_coerce_summary_by_profile_row(row) for row in _read_csv_rows(summary_by_profile_path)]

    phase_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        phase_groups[(str(row["profile_name"]), str(row["phase"]))].append(row)

    updated_profile_rows: list[dict[str, Any]] = []
    for row in profile_rows:
        training_rates = _observed_failure_rates(phase_groups.get((str(row["profile_name"]), "training"), []))
        evaluation_rates = _observed_failure_rates(phase_groups.get((str(row["profile_name"]), "evaluation"), []))
        updated_profile_rows.append(
            {
                **row,
                "train_step_limit_rate": training_rates["step_limit_rate"],
                "train_no_progress_rate": training_rates["no_progress_rate"],
                "train_wall_contact_rate": training_rates["wall_contact_rate"],
                "train_R_m_observed": training_rates["R_m_observed"],
                "eval_step_limit_rate": evaluation_rates["step_limit_rate"],
                "eval_no_progress_rate": evaluation_rates["no_progress_rate"],
                "eval_wall_contact_rate": evaluation_rates["wall_contact_rate"],
                "eval_R_m_observed": evaluation_rates["R_m_observed"],
            }
        )

    if updated_profile_rows:
        write_csv(summary_by_profile_path, updated_profile_rows, list(updated_profile_rows[0].keys()))

    aggregated_rows = _aggregate_summary_by_agent_condition_rows_from_profile_rows(updated_profile_rows)
    if aggregated_rows:
        write_csv(
            os.path.join(out_dir, "summary_by_agent_condition.csv"),
            aggregated_rows,
            list(aggregated_rows[0].keys()),
        )


def _break_even_label(is_helpful: bool) -> str:
    return "helpful" if is_helpful else "harmful"


def _observed_outcome_label(is_solved: bool) -> str:
    return "solved" if is_solved else "failed"


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None else None


def _build_break_even_seed_rows(
    profile_rows: list[dict[str, Any]],
    *,
    lambda_failure: float,
) -> list[dict[str, Any]]:
    primitive_lookup = {
        (str(row["condition"]), str(row["run_id"])): row
        for row in profile_rows
        if str(row["agent_type"]) == "DQN_LSTM_PRIMITIVE"
    }
    rows: list[dict[str, Any]] = []
    for row in sorted(
        profile_rows,
        key=lambda r: (str(r["condition"]), str(r["agent_type"]), int(r.get("seed") or 0)),
    ):
        agent = str(row["agent_type"])
        if agent == "DQN_LSTM_PRIMITIVE":
            continue
        primitive = primitive_lookup.get((str(row["condition"]), str(row["run_id"])))
        if primitive is None:
            continue
        primitive_steps = float(primitive.get("train_mean_steps") or 0.0)
        primitive_decisions = float(primitive.get("train_mean_decisions") or 0.0)
        macro_steps = float(row.get("train_mean_steps") or 0.0)
        macro_decisions = float(row.get("train_mean_decisions") or 0.0)
        compression = (primitive_decisions / macro_decisions) if macro_decisions > 0 else None
        overhead = ((macro_steps - primitive_steps) / primitive_steps) if primitive_steps > 0 else None
        r_m_proxy = 1.0 - float(row.get("train_success_rate") or 0.0)
        r_m_observed = float(row.get("train_R_m_observed") or 0.0)
        threshold_proxy = (1.0 + overhead + r_m_proxy) if overhead is not None else None
        threshold_observed = (1.0 + overhead + (lambda_failure * r_m_observed)) if overhead is not None else None
        predicted_proxy = bool(compression is not None and threshold_proxy is not None and compression > threshold_proxy)
        predicted_observed = bool(
            compression is not None and threshold_observed is not None and compression > threshold_observed
        )
        observed_solved = float(row.get("eval_success_rate") or 0.0) > 0.5
        rows.append(
            {
                "condition": row["condition"],
                "agent_type": agent,
                "run_id": row["run_id"],
                "seed": row["seed"],
                "train_success_rate": row.get("train_success_rate"),
                "eval_success_rate": row.get("eval_success_rate"),
                "primitive_train_mean_steps": primitive_steps,
                "macro_train_mean_steps": macro_steps,
                "primitive_train_mean_decisions": primitive_decisions,
                "macro_train_mean_decisions": macro_decisions,
                "compression_ratio_C": _rounded(compression),
                "step_overhead_O": _rounded(overhead),
                "step_limit_rate": row.get("train_step_limit_rate"),
                "no_progress_rate": row.get("train_no_progress_rate"),
                "wall_contact_rate": row.get("train_wall_contact_rate"),
                "R_m_proxy": _rounded(r_m_proxy),
                "R_m_observed": row.get("train_R_m_observed"),
                "lambda_failure": _rounded(lambda_failure),
                "threshold_proxy": _rounded(threshold_proxy),
                "threshold_observed": _rounded(threshold_observed),
                "predicted_proxy": _break_even_label(predicted_proxy),
                "predicted_observed": _break_even_label(predicted_observed),
                "observed_solved": _observed_outcome_label(observed_solved),
                "classification_changed": int(predicted_proxy != predicted_observed),
                "note": LOOPS_NOT_DIRECTLY_LOGGED_NOTE,
            }
        )
    return rows


def _build_break_even_agent_condition_rows(
    aggregated_rows: list[dict[str, Any]],
    *,
    lambda_failure: float,
) -> list[dict[str, Any]]:
    primitive_lookup = {
        str(row["condition"]): row
        for row in aggregated_rows
        if str(row["agent_type"]) == "DQN_LSTM_PRIMITIVE"
    }
    rows: list[dict[str, Any]] = []
    for row in sorted(
        aggregated_rows,
        key=lambda r: (str(r["condition"]), str(r["agent_type"])),
    ):
        agent = str(row["agent_type"])
        if agent == "DQN_LSTM_PRIMITIVE":
            continue
        primitive = primitive_lookup.get(str(row["condition"]))
        if primitive is None:
            continue
        primitive_steps = float(primitive.get("train_mean_steps_mean") or 0.0)
        primitive_decisions = float(primitive.get("train_mean_decisions_mean") or 0.0)
        macro_steps = float(row.get("train_mean_steps_mean") or 0.0)
        macro_decisions = float(row.get("train_mean_decisions_mean") or 0.0)
        compression = (primitive_decisions / macro_decisions) if macro_decisions > 0 else None
        overhead = ((macro_steps - primitive_steps) / primitive_steps) if primitive_steps > 0 else None
        r_m_proxy = 1.0 - float(row.get("train_success_rate_mean") or 0.0)
        r_m_observed = float(row.get("train_R_m_observed_mean") or 0.0)
        threshold_proxy = (1.0 + overhead + r_m_proxy) if overhead is not None else None
        threshold_observed = (1.0 + overhead + (lambda_failure * r_m_observed)) if overhead is not None else None
        predicted_proxy = bool(compression is not None and threshold_proxy is not None and compression > threshold_proxy)
        predicted_observed = bool(
            compression is not None and threshold_observed is not None and compression > threshold_observed
        )
        observed_solved = float(row.get("eval_success_rate_mean") or 0.0) > 0.5
        rows.append(
            {
                "condition": row["condition"],
                "agent_type": agent,
                "n_profiles": row["n_profiles"],
                "train_success_rate_mean": row.get("train_success_rate_mean"),
                "eval_success_rate_mean": row.get("eval_success_rate_mean"),
                "primitive_train_mean_steps": primitive_steps,
                "macro_train_mean_steps": macro_steps,
                "primitive_train_mean_decisions": primitive_decisions,
                "macro_train_mean_decisions": macro_decisions,
                "compression_ratio_C": _rounded(compression),
                "step_overhead_O": _rounded(overhead),
                "step_limit_rate_mean": row.get("train_step_limit_rate_mean"),
                "step_limit_rate_std": row.get("train_step_limit_rate_std"),
                "no_progress_rate_mean": row.get("train_no_progress_rate_mean"),
                "no_progress_rate_std": row.get("train_no_progress_rate_std"),
                "wall_contact_rate_mean": row.get("train_wall_contact_rate_mean"),
                "wall_contact_rate_std": row.get("train_wall_contact_rate_std"),
                "R_m_proxy": _rounded(r_m_proxy),
                "R_m_observed_mean": row.get("train_R_m_observed_mean"),
                "R_m_observed_std": row.get("train_R_m_observed_std"),
                "lambda_failure": _rounded(lambda_failure),
                "threshold_proxy": _rounded(threshold_proxy),
                "threshold_observed": _rounded(threshold_observed),
                "predicted_proxy": _break_even_label(predicted_proxy),
                "predicted_observed": _break_even_label(predicted_observed),
                "observed_solved": _observed_outcome_label(observed_solved),
                "classification_changed": int(predicted_proxy != predicted_observed),
                "note": LOOPS_NOT_DIRECTLY_LOGGED_NOTE,
            }
        )
    return rows


def write_break_even_tables_from_existing_exports(
    out_dir: str,
    *,
    lambda_failure: float = DEFAULT_LAMBDA_FAILURE,
) -> None:
    summary_by_profile_path = os.path.join(out_dir, "summary_by_profile.csv")
    summary_by_agent_condition_path = os.path.join(out_dir, "summary_by_agent_condition.csv")
    if not os.path.exists(summary_by_profile_path):
        raise FileNotFoundError(f"Missing summary export: {summary_by_profile_path}")
    if not os.path.exists(summary_by_agent_condition_path):
        raise FileNotFoundError(f"Missing aggregated summary export: {summary_by_agent_condition_path}")

    profile_rows = [_coerce_summary_by_profile_row(row) for row in _read_csv_rows(summary_by_profile_path)]
    aggregated_rows = [
        _coerce_summary_by_agent_condition_row(row)
        for row in _read_csv_rows(summary_by_agent_condition_path)
    ]

    by_seed_rows = _build_break_even_seed_rows(profile_rows, lambda_failure=lambda_failure)
    if by_seed_rows:
        write_csv(
            os.path.join(out_dir, "break_even_table_by_seed.csv"),
            by_seed_rows,
            list(by_seed_rows[0].keys()),
        )

    by_agent_condition_rows = _build_break_even_agent_condition_rows(
        aggregated_rows,
        lambda_failure=lambda_failure,
    )
    if by_agent_condition_rows:
        write_csv(
            os.path.join(out_dir, "break_even_table_by_agent_condition.csv"),
            by_agent_condition_rows,
            list(by_agent_condition_rows[0].keys()),
        )


def _config_row(agent: str, condition_id: str, condition: ConditionSpec, *, replay_warmup_policy: str) -> dict[str, Any]:
    config = build_config_for_agent(
        agent,
        condition,
        replay_warmup_policy=replay_warmup_policy,
    )
    return {
        "agent_type": agent,
        "condition": condition_id,
        "bot_type": "DQNBot",
        "learning_rate": getattr(config, "learningRate", None),
        "discount_factor": getattr(config, "discountFactor", None),
        "epsilon_start": getattr(config, "epsilonStart", None),
        "epsilon_end": getattr(config, "epsilonEnd", None),
        "epsilon_decay_steps": getattr(config, "epsilonDecaySteps", None),
        "replay_warmup_steps": getattr(config, "replayWarmupSteps", None),
        "replay_capacity": getattr(config, "replayCapacity", None),
        "batch_size": getattr(config, "batchSize", None),
        "train_frequency": getattr(config, "trainFrequency", None),
        "target_update_frequency": getattr(config, "targetUpdateFrequency", None),
        "hidden_size": getattr(config, "hiddenSize", None),
        "use_shared_comparison_state": getattr(config, "useSharedComparisonState", None),
        "use_rich_encoding": getattr(config, "useRichEncoding", None),
        "use_lstm_policy": getattr(config, "useLstmPolicy", None),
        "lstm_sequence_length": getattr(config, "lstmSequenceLength", None),
        "lstm_hidden_size": getattr(config, "lstmHiddenSize", None),
        "use_macro_actions": getattr(config, "useMacroActions", None),
        "macro_option_set": getattr(config, "macroOptionSet", None),
        "use_hierarchical_policy": getattr(config, "useHierarchicalPolicy", None),
        "immediate_reversal_penalty": getattr(config, "immediateReversalPenalty", None),
        "repeat_visit_penalty_scale": getattr(config, "repeatVisitPenaltyScale", None),
        "no_progress_penalty": getattr(config, "noProgressPenalty", None),
        "step_limit_step_coeff": getattr(config, "stepLimitStepCoeff", None),
        "step_limit_area_coeff": getattr(config, "stepLimitAreaCoeff", None),
        "step_limit_min": getattr(config, "stepLimitMin", None),
        "step_limit_max": getattr(config, "stepLimitMax", None),
        "step_limit_penalty": getattr(config, "stepLimitPenalty", None),
    }


def write_final_report_tables(
    out_dir: str,
    plan_raw: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    conditions = plan_raw.get("conditions", {})
    agents_in_plan = plan_raw.get("agents", [])
    replay_warmup_policy = str(plan_raw.get("dqn_replay_warmup_policy", "disabled"))

    maze_table = [
        {
            "condition": condition_id,
            "mode": raw.get("mode"),
            "width": raw.get("width"),
            "height": raw.get("height"),
            "training_episodes": raw.get("training_episodes"),
            "evaluation_episodes": raw.get("evaluation_episodes"),
            "evaluation_episodes_per_pool_maze": raw.get("evaluation_episodes_per_pool_maze"),
            "maze_id": raw.get("maze_id") or raw.get("maze_id_prefix"),
            "maze_path": raw.get("maze_path"),
            "maze_paths": raw.get("maze_paths"),
            "training_maze_paths": raw.get("training_maze_paths"),
            "evaluation_maze_paths": raw.get("evaluation_maze_paths"),
            "pool_size": raw.get("pool_size"),
        }
        for condition_id, raw in conditions.items()
    ]

    agent_config_rows: list[dict[str, Any]] = []
    for condition_id, raw in conditions.items():
        condition = _condition_from_raw(condition_id, raw)
        for agent in agents_in_plan:
            agent_config_rows.append(
                _config_row(
                    agent,
                    condition_id,
                    condition,
                    replay_warmup_policy=replay_warmup_policy,
                )
            )

    reward_table = [{"event": event, "value": value} for event, value in SHARED_REWARD_MODIFIERS.items()]

    final_evaluation_table: list[dict[str, Any]] = []
    for summary in summaries:
        evaluation = summary.get("evaluation", {})
        final_evaluation_table.append(
            {
                "profile_name": summary["profile_name"],
                "agent_type": summary["agent_type"],
                "condition": summary["condition"],
                "run_id": summary["run_id"],
                "eval_success_rate": evaluation.get("success_rate"),
                "eval_mean_reward": evaluation.get("mean_reward"),
                "eval_std_reward": evaluation.get("std_reward"),
                "eval_mean_steps": evaluation.get("mean_steps"),
                "eval_mean_path_inefficiency_ratio": evaluation.get("mean_path_inefficiency_ratio"),
            }
        )

    dqn_diagnostics_table: list[dict[str, Any]] = []
    for summary in summaries:
        diagnostics = summary.get("dqn_diagnostics", {})
        dqn_diagnostics_table.append(
            {
                "profile_name": summary["profile_name"],
                "agent_type": summary["agent_type"],
                "condition": summary["condition"],
                "run_id": summary["run_id"],
                **diagnostics,
            }
        )

    write_json(
        os.path.join(out_dir, "final_report_tables.json"),
        {
            "protocol": {
                "dqn_replay_warmup_policy": replay_warmup_policy,
                "lambda_failure": DEFAULT_LAMBDA_FAILURE,
                "break_even_R_m_definition": (
                    "mean(step_limit_failure) + mean(no_progress_failure) + mean(wall_contact_failure)"
                ),
                "notes": [LOOPS_NOT_DIRECTLY_LOGGED_NOTE],
            },
            "maze_table": maze_table,
            "agent_config_table": agent_config_rows,
            "reward_table": reward_table,
            "final_evaluation_table": final_evaluation_table,
            "dqn_diagnostics_table": dqn_diagnostics_table,
        },
    )
