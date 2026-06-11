from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import pytest

from services.research.guiSupport import (
    get_research_preset,
    get_static_research_presets,
    load_research_summary,
)
from services.research.makeFigures import _resolve_experiment_dir, generate_report_figures
from services.research.metrics import EPISODES_CSV_COLUMNS
from services.research.plan import ConditionSpec, ExperimentPlan, PlanValidationError
from services.research.profileBuilder import (
    build_config_for_agent,
    build_dqn_flat_config,
    build_dqn_local_lstm_config,
    build_dqn_lstm_macro_config,
    build_shared_reward_config,
)
from services.research.writers import (
    refresh_observed_failure_metrics_from_episodes,
    write_break_even_tables_from_existing_exports,
    write_summary_by_agent_condition,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_plan(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_static_research_presets_are_macro_only() -> None:
    presets = get_static_research_presets()

    assert [preset.preset_id for preset in presets] == ["macro_action_quality"]
    assert presets[0].plan_path == ROOT / "research" / "plans" / "lstm_macro_plan.json"
    assert presets[0].output_dir == ROOT / "research" / "results" / "lstm_macro_run"


def test_macro_action_quality_summary_matches_supported_plan() -> None:
    preset = get_research_preset("macro_action_quality")
    summary = load_research_summary(preset)

    assert summary.experiment_name == "lstm_macro_action_quality"
    assert summary.curriculum_training == ()
    assert summary.curriculum_evaluation == ()
    assert summary.total_profiles == 48
    assert summary.total_training_episodes == 156000
    assert summary.total_evaluation_episodes == 6600
    assert any("POOL_MED_HELDOUT | pool 15x15" in line for line in summary.condition_lines)


def test_removed_curriculum_preset_is_rejected() -> None:
    with pytest.raises(KeyError, match="macro_action_curriculum"):
        get_research_preset("macro_action_curriculum")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "experiment_name": "x",
                "branch": "main",
                "seeds": [1],
                "seed_labels": ["S0"],
                "agents": ["DQN_LSTM_PRIMITIVE"],
                "conditions": {
                    "FX": {
                        "mode": "fixed",
                        "width": 8,
                        "height": 8,
                        "training_episodes": 10,
                        "evaluation_episodes": 5,
                        "maze_id": "fixed_small",
                    }
                },
            },
            "maze_path",
        ),
        (
            {
                "experiment_name": "x",
                "branch": "main",
                "seeds": [1],
                "seed_labels": ["S0"],
                "agents": ["DQN_LSTM_PRIMITIVE"],
                "conditions": {
                    "POOL": {
                        "mode": "pool",
                        "width": 15,
                        "height": 15,
                        "pool_size": 2,
                        "training_episodes": 10,
                        "evaluation_episodes": 3,
                        "evaluation_episodes_per_pool_maze": 2,
                        "maze_id_prefix": "pool",
                        "maze_paths": ["a.json", "b.json"],
                    }
                },
            },
            r"evaluation pool size \* evaluation_episodes_per_pool_maze",
        ),
    ],
)
def test_plan_validation_rejects_invalid_fixed_and_pool_inputs(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    plan_path = _write_plan(tmp_path, "invalid_plan.json", payload)

    with pytest.raises(PlanValidationError, match=message):
        ExperimentPlan.load(str(plan_path))


def test_plan_allows_split_pool_with_heldout_evaluation_subset(tmp_path: Path) -> None:
    plan_path = _write_plan(
        tmp_path,
        "heldout_pool_plan.json",
        {
            "experiment_name": "x",
            "branch": "main",
            "seeds": [1],
            "seed_labels": ["S0"],
            "agents": ["DQN_LSTM_PRIMITIVE"],
            "conditions": {
                "POOL": {
                    "mode": "pool",
                    "width": 15,
                    "height": 15,
                    "pool_size": 3,
                    "training_episodes": 12,
                    "evaluation_episodes": 4,
                    "evaluation_episodes_per_pool_maze": 2,
                    "maze_id_prefix": "pool",
                    "training_maze_paths": ["a.json", "b.json", "c.json"],
                    "evaluation_maze_paths": ["d.json", "e.json"],
                }
            },
        },
    )

    plan = ExperimentPlan.load(str(plan_path))
    pool = plan.conditions["POOL"]

    assert pool.pool_size == 3
    assert len(pool.training_maze_paths) == 3
    assert len(pool.evaluation_maze_paths) == 2
    assert pool.evaluation_episodes == 4


def test_research_config_builders_cover_supported_macro_variants() -> None:
    condition = ConditionSpec(
        condition_id="FX_MED",
        mode="fixed",
        width=15,
        height=15,
        training_episodes=100,
        evaluation_episodes=20,
        maze_id="fixed_medium",
        maze_path="fixed_medium.json",
    )

    flat = build_dqn_flat_config(condition, replay_warmup_policy="budget_scaled")
    primitive = build_dqn_local_lstm_config(condition, replay_warmup_policy="disabled")
    naive = build_dqn_lstm_macro_config(
        condition,
        replay_warmup_policy="disabled",
        macro_option_set="naive",
    )
    momentum = build_config_for_agent(
        "DQN_LSTM_MOMENTUM_MACROS",
        condition,
        replay_warmup_policy="disabled",
    )
    reward_config = build_shared_reward_config()

    assert flat.replayWarmupSteps == 2000
    assert primitive.useLstmPolicy is True
    assert primitive.useMacroActions is False
    assert naive.useMacroActions is True
    assert naive.macroOptionSet == "naive"
    assert momentum.macroOptionSet == "momentum"
    assert reward_config.rewardModifiers["move_in_optimal_path"] == "0"
    assert reward_config.rewardModifiers["new_tile_visited"] == "0"
    assert reward_config.usePotentialShaping is True

    with pytest.raises(ValueError, match="Unsupported research-mode agent type"):
        build_config_for_agent("DQN_LOCAL", condition, replay_warmup_policy="disabled")


def test_summary_by_agent_condition_aggregates_seed_runs(tmp_path: Path) -> None:
    summaries = [
        {
            "profile_name": "DQN_LSTM_PRIMITIVE_FX_S0",
            "agent_type": "DQN_LSTM_PRIMITIVE",
            "bot_type": "DQNBot",
            "condition": "FX",
            "run_id": "S0",
            "seed": 1,
            "training_episodes": 10,
            "evaluation_episodes": 5,
            "training": {
                "success_rate": 0.2,
                "mean_reward": 10.0,
                "mean_steps": 40.0,
                "mean_decisions": 40.0,
                "mean_action_duration": 1.0,
                "mean_option_step_fraction": 0.0,
                "step_limit_rate": 0.1,
                "no_progress_rate": 0.2,
                "wall_contact_rate": 0.3,
                "R_m_observed": 0.6,
                "first_success_episode": 7,
                "final_50_success_rate": 0.3,
            },
            "evaluation": {
                "success_rate": 0.4,
                "mean_reward": 20.0,
                "mean_steps": 30.0,
                "mean_decisions": 30.0,
                "mean_action_duration": 1.0,
                "mean_option_step_fraction": 0.0,
                "step_limit_rate": 0.0,
                "no_progress_rate": 0.4,
                "wall_contact_rate": 0.2,
                "R_m_observed": 0.6,
                "mean_path_inefficiency_ratio": 1.5,
                "mean_wall_hits": 3.0,
            },
        },
        {
            "profile_name": "DQN_LSTM_PRIMITIVE_FX_S1",
            "agent_type": "DQN_LSTM_PRIMITIVE",
            "bot_type": "DQNBot",
            "condition": "FX",
            "run_id": "S1",
            "seed": 2,
            "training_episodes": 10,
            "evaluation_episodes": 5,
            "training": {
                "success_rate": 0.6,
                "mean_reward": 30.0,
                "mean_steps": 20.0,
                "mean_decisions": 20.0,
                "mean_action_duration": 1.0,
                "mean_option_step_fraction": 0.0,
                "step_limit_rate": 0.3,
                "no_progress_rate": 0.1,
                "wall_contact_rate": 0.0,
                "R_m_observed": 0.4,
                "first_success_episode": 3,
                "final_50_success_rate": 0.7,
            },
            "evaluation": {
                "success_rate": 0.8,
                "mean_reward": 40.0,
                "mean_steps": 10.0,
                "mean_decisions": 10.0,
                "mean_action_duration": 1.0,
                "mean_option_step_fraction": 0.0,
                "step_limit_rate": 0.2,
                "no_progress_rate": 0.0,
                "wall_contact_rate": 0.2,
                "R_m_observed": 0.4,
                "mean_path_inefficiency_ratio": 1.1,
                "mean_wall_hits": 1.0,
            },
        },
    ]

    write_summary_by_agent_condition(str(tmp_path), summaries)

    with (tmp_path / "summary_by_agent_condition.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["agentType"] == "DQN_LSTM_PRIMITIVE"
    assert row["condition"] == "FX"
    assert row["nProfiles"] == "2"
    assert math.isclose(float(row["trainSuccessRateMean"]), 0.4)
    assert math.isclose(float(row["trainRMObservedMean"]), 0.5)
    assert math.isclose(float(row["evalNoProgressRateMean"]), 0.2)
    assert math.isclose(float(row["evalMeanRewardMean"]), 30.0)
    assert math.isclose(float(row["trainFirstSuccessEpisodeMean"]), 5.0)


def test_refresh_observed_failure_metrics_and_break_even_exports(tmp_path: Path) -> None:
    episode_rows = [
        {
            "experiment_name": "exp",
            "profile_name": "DQN_LSTM_PRIMITIVE_FX_S0",
            "agent_type": "DQN_LSTM_PRIMITIVE",
            "bot_type": "DQNBot",
            "condition": "FX",
            "maze_mode": "fixed",
            "maze_size": "8x8",
            "maze_id": "maze",
            "run_id": "S0",
            "seed": 1,
            "phase": "training",
            "episode": 1,
            "global_episode": 1,
            "success": 1,
            "outcome": "goal_reached",
            "total_reward": 10.0,
            "steps": 10,
            "cumulative_env_steps": "",
            "cumulative_decisions": "",
            "optimal_steps": 8,
            "path_inefficiency_ratio": 1.25,
            "decisions": 10,
            "mean_action_duration": 1.0,
            "option_selections": 0,
            "option_steps": 0,
            "option_step_fraction": 0.0,
            "times_hit_wall": 0,
            "step_limit_failure": 0,
            "no_progress_failure": 0,
            "wall_contact_failure": 0,
            "times_hit_enemy": 0,
            "enemy_kills": 0,
            "q_table_states": "",
            "replay_size": "",
            "warmup_required": "",
            "warmup_complete": "",
            "training_updates": "",
            "epsilon": "",
            "eval_epsilon": "",
            "epsilon_is_manual": 0,
            "loss": "",
            "elapsed_seconds": 0.1,
            "timestamp": "2026-01-01T00:00:00",
        },
        {
            "experiment_name": "exp",
            "profile_name": "DQN_LSTM_PRIMITIVE_FX_S0",
            "agent_type": "DQN_LSTM_PRIMITIVE",
            "bot_type": "DQNBot",
            "condition": "FX",
            "maze_mode": "fixed",
            "maze_size": "8x8",
            "maze_id": "maze",
            "run_id": "S0",
            "seed": 1,
            "phase": "evaluation",
            "episode": 1,
            "global_episode": 2,
            "success": 1,
            "outcome": "goal_reached",
            "total_reward": 10.0,
            "steps": 10,
            "cumulative_env_steps": "",
            "cumulative_decisions": "",
            "optimal_steps": 8,
            "path_inefficiency_ratio": 1.25,
            "decisions": 10,
            "mean_action_duration": 1.0,
            "option_selections": 0,
            "option_steps": 0,
            "option_step_fraction": 0.0,
            "times_hit_wall": 0,
            "step_limit_failure": 0,
            "no_progress_failure": 0,
            "wall_contact_failure": 0,
            "times_hit_enemy": 0,
            "enemy_kills": 0,
            "q_table_states": "",
            "replay_size": "",
            "warmup_required": "",
            "warmup_complete": "",
            "training_updates": "",
            "epsilon": "",
            "eval_epsilon": "",
            "epsilon_is_manual": 0,
            "loss": "",
            "elapsed_seconds": 0.1,
            "timestamp": "2026-01-01T00:00:01",
        },
        {
            "experiment_name": "exp",
            "profile_name": "DQN_LSTM_MOMENTUM_MACROS_FX_S0",
            "agent_type": "DQN_LSTM_MOMENTUM_MACROS",
            "bot_type": "DQNBot",
            "condition": "FX",
            "maze_mode": "fixed",
            "maze_size": "8x8",
            "maze_id": "maze",
            "run_id": "S0",
            "seed": 1,
            "phase": "training",
            "episode": 1,
            "global_episode": 3,
            "success": 0,
            "outcome": "no_progress",
            "total_reward": -10.0,
            "steps": 14,
            "cumulative_env_steps": "",
            "cumulative_decisions": "",
            "optimal_steps": 8,
            "path_inefficiency_ratio": 1.75,
            "decisions": 6,
            "mean_action_duration": 2.3333,
            "option_selections": 1,
            "option_steps": 8,
            "option_step_fraction": 0.5714,
            "times_hit_wall": 0,
            "step_limit_failure": 0,
            "no_progress_failure": 1,
            "wall_contact_failure": 0,
            "times_hit_enemy": 0,
            "enemy_kills": 0,
            "q_table_states": "",
            "replay_size": "",
            "warmup_required": "",
            "warmup_complete": "",
            "training_updates": "",
            "epsilon": "",
            "eval_epsilon": "",
            "epsilon_is_manual": 0,
            "loss": "",
            "elapsed_seconds": 0.1,
            "timestamp": "2026-01-01T00:00:02",
        },
        {
            "experiment_name": "exp",
            "profile_name": "DQN_LSTM_MOMENTUM_MACROS_FX_S0",
            "agent_type": "DQN_LSTM_MOMENTUM_MACROS",
            "bot_type": "DQNBot",
            "condition": "FX",
            "maze_mode": "fixed",
            "maze_size": "8x8",
            "maze_id": "maze",
            "run_id": "S0",
            "seed": 1,
            "phase": "training",
            "episode": 2,
            "global_episode": 4,
            "success": 1,
            "outcome": "goal_reached",
            "total_reward": 10.0,
            "steps": 10,
            "cumulative_env_steps": "",
            "cumulative_decisions": "",
            "optimal_steps": 8,
            "path_inefficiency_ratio": 1.25,
            "decisions": 5,
            "mean_action_duration": 2.0,
            "option_selections": 1,
            "option_steps": 6,
            "option_step_fraction": 0.6,
            "times_hit_wall": 1,
            "step_limit_failure": 0,
            "no_progress_failure": 0,
            "wall_contact_failure": 1,
            "times_hit_enemy": 0,
            "enemy_kills": 0,
            "q_table_states": "",
            "replay_size": "",
            "warmup_required": "",
            "warmup_complete": "",
            "training_updates": "",
            "epsilon": "",
            "eval_epsilon": "",
            "epsilon_is_manual": 0,
            "loss": "",
            "elapsed_seconds": 0.1,
            "timestamp": "2026-01-01T00:00:03",
        },
        {
            "experiment_name": "exp",
            "profile_name": "DQN_LSTM_MOMENTUM_MACROS_FX_S0",
            "agent_type": "DQN_LSTM_MOMENTUM_MACROS",
            "bot_type": "DQNBot",
            "condition": "FX",
            "maze_mode": "fixed",
            "maze_size": "8x8",
            "maze_id": "maze",
            "run_id": "S0",
            "seed": 1,
            "phase": "evaluation",
            "episode": 1,
            "global_episode": 5,
            "success": 0,
            "outcome": "step_limit",
            "total_reward": -10.0,
            "steps": 14,
            "cumulative_env_steps": "",
            "cumulative_decisions": "",
            "optimal_steps": 8,
            "path_inefficiency_ratio": 1.75,
            "decisions": 6,
            "mean_action_duration": 2.3333,
            "option_selections": 1,
            "option_steps": 8,
            "option_step_fraction": 0.5714,
            "times_hit_wall": 0,
            "step_limit_failure": 1,
            "no_progress_failure": 0,
            "wall_contact_failure": 0,
            "times_hit_enemy": 0,
            "enemy_kills": 0,
            "q_table_states": "",
            "replay_size": "",
            "warmup_required": "",
            "warmup_complete": "",
            "training_updates": "",
            "epsilon": "",
            "eval_epsilon": "",
            "epsilon_is_manual": 0,
            "loss": "",
            "elapsed_seconds": 0.1,
            "timestamp": "2026-01-01T00:00:04",
        },
    ]
    with (tmp_path / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODES_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(episode_rows)

    summary_rows = [
        {
            "profile_name": "DQN_LSTM_PRIMITIVE_FX_S0",
            "agent_type": "DQN_LSTM_PRIMITIVE",
            "bot_type": "DQNBot",
            "condition": "FX",
            "run_id": "S0",
            "seed": 1,
            "training_episodes": 1,
            "evaluation_episodes": 1,
            "train_success_rate": 1.0,
            "train_mean_reward": 10.0,
            "train_std_reward": 0.0,
            "train_mean_steps": 10.0,
            "train_mean_decisions": 9.0,
            "train_mean_action_duration": 1.0,
            "train_mean_option_selections": 0.0,
            "train_mean_option_step_fraction": 0.0,
            "train_first_success": 1,
            "train_final_50_success_rate": 1.0,
            "train_final_50_mean_reward": 10.0,
            "eval_success_rate": 1.0,
            "eval_mean_reward": 10.0,
            "eval_std_reward": 0.0,
            "eval_mean_steps": 10.0,
            "eval_std_steps": 0.0,
            "eval_mean_decisions": 9.0,
            "eval_mean_action_duration": 1.0,
            "eval_mean_option_selections": 0.0,
            "eval_mean_option_step_fraction": 0.0,
            "eval_mean_path_inefficiency_ratio": 1.25,
            "eval_mean_wall_hits": 0.0,
            "macro_option_set": "naive",
            "total_option_selections": 0,
        },
        {
            "profile_name": "DQN_LSTM_MOMENTUM_MACROS_FX_S0",
            "agent_type": "DQN_LSTM_MOMENTUM_MACROS",
            "bot_type": "DQNBot",
            "condition": "FX",
            "run_id": "S0",
            "seed": 1,
            "training_episodes": 2,
            "evaluation_episodes": 1,
            "train_success_rate": 0.5,
            "train_mean_reward": 0.0,
            "train_std_reward": 10.0,
            "train_mean_steps": 12.0,
            "train_mean_decisions": 5.0,
            "train_mean_action_duration": 2.1667,
            "train_mean_option_selections": 1.0,
            "train_mean_option_step_fraction": 0.5857,
            "train_first_success": 2,
            "train_final_50_success_rate": 0.5,
            "train_final_50_mean_reward": 0.0,
            "eval_success_rate": 0.0,
            "eval_mean_reward": -10.0,
            "eval_std_reward": 0.0,
            "eval_mean_steps": 14.0,
            "eval_std_steps": 0.0,
            "eval_mean_decisions": 6.0,
            "eval_mean_action_duration": 2.3333,
            "eval_mean_option_selections": 1.0,
            "eval_mean_option_step_fraction": 0.5714,
            "eval_mean_path_inefficiency_ratio": 1.75,
            "eval_mean_wall_hits": 0.0,
            "macro_option_set": "momentum",
            "total_option_selections": 3,
        },
    ]
    with (tmp_path / "summary_by_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    refresh_observed_failure_metrics_from_episodes(str(tmp_path))
    write_break_even_tables_from_existing_exports(str(tmp_path))

    with (tmp_path / "summary_by_profile.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    macro_row = next(row for row in rows if row["agentType"] == "DQN_LSTM_MOMENTUM_MACROS")
    assert math.isclose(float(macro_row["trainStepLimitRate"]), 0.0)
    assert math.isclose(float(macro_row["trainNoProgressRate"]), 0.5)
    assert math.isclose(float(macro_row["trainWallContactRate"]), 0.5)
    assert math.isclose(float(macro_row["trainRMObserved"]), 1.0)

    with (tmp_path / "break_even_table_by_agent_condition.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["agentType"] == "DQN_LSTM_MOMENTUM_MACROS"
    assert math.isclose(float(row["rMProxy"]), 0.5)
    assert math.isclose(float(row["rMObservedMean"]), 1.0)
    assert math.isclose(float(row["thresholdProxy"]), 1.7)
    assert math.isclose(float(row["thresholdObserved"]), 2.2)
    assert row["predictedProxy"] == "helpful"
    assert row["predictedObserved"] == "harmful"
    assert row["classificationChanged"] == "1"
    assert row["note"] == "Loops are not directly logged in the current experiment."


def test_generate_report_figures_uses_current_aggregates(tmp_path: Path) -> None:
    summary_rows = [
        {
            "agent_type": "DQN_LSTM_PRIMITIVE",
            "condition": "FX_SMALL",
            "n_profiles": 3,
            "train_success_rate_mean": 0.84,
            "train_success_rate_std": 0.01,
            "train_mean_reward_mean": 0.0,
            "train_mean_reward_std": 0.0,
            "train_mean_steps_mean": 28.0,
            "train_mean_steps_std": 0.0,
            "train_mean_decisions_mean": 28.0,
            "train_mean_decisions_std": 0.0,
            "train_mean_action_duration_mean": 1.0,
            "train_mean_option_step_fraction_mean": 0.0,
            "train_first_success_episode_mean": 1.0,
            "train_first_success_episode_std": 0.0,
            "train_final_50_success_rate_mean": 1.0,
            "train_final_50_success_rate_std": 0.0,
            "eval_success_rate_mean": 1.0,
            "eval_success_rate_std": 0.0,
            "eval_mean_reward_mean": 0.0,
            "eval_mean_reward_std": 0.0,
            "eval_mean_steps_mean": 7.0,
            "eval_mean_steps_std": 0.0,
            "eval_mean_decisions_mean": 7.0,
            "eval_mean_decisions_std": 0.0,
            "eval_mean_action_duration_mean": 1.0,
            "eval_mean_option_step_fraction_mean": 0.0,
            "eval_mean_path_inefficiency_ratio_mean": 0.8,
            "eval_mean_path_inefficiency_ratio_std": 0.0,
            "eval_mean_wall_hits_mean": 0.0,
            "eval_mean_wall_hits_std": 0.0,
        },
        {
            "agent_type": "DQN_LSTM_NAIVE_MACROS",
            "condition": "FX_SMALL",
            "n_profiles": 3,
            "train_success_rate_mean": 0.93,
            "train_success_rate_std": 0.01,
            "train_mean_reward_mean": 0.0,
            "train_mean_reward_std": 0.0,
            "train_mean_steps_mean": 27.0,
            "train_mean_steps_std": 0.0,
            "train_mean_decisions_mean": 20.0,
            "train_mean_decisions_std": 0.0,
            "train_mean_action_duration_mean": 1.35,
            "train_mean_option_step_fraction_mean": 0.0,
            "train_first_success_episode_mean": 1.0,
            "train_first_success_episode_std": 0.0,
            "train_final_50_success_rate_mean": 1.0,
            "train_final_50_success_rate_std": 0.0,
            "eval_success_rate_mean": 1.0,
            "eval_success_rate_std": 0.0,
            "eval_mean_reward_mean": 0.0,
            "eval_mean_reward_std": 0.0,
            "eval_mean_steps_mean": 7.0,
            "eval_mean_steps_std": 0.0,
            "eval_mean_decisions_mean": 4.0,
            "eval_mean_decisions_std": 0.0,
            "eval_mean_action_duration_mean": 1.75,
            "eval_mean_option_step_fraction_mean": 0.0,
            "eval_mean_path_inefficiency_ratio_mean": 0.8,
            "eval_mean_path_inefficiency_ratio_std": 0.0,
            "eval_mean_wall_hits_mean": 0.0,
            "eval_mean_wall_hits_std": 0.0,
        },
        {
            "agent_type": "DQN_LSTM_ASTAR_MACROS",
            "condition": "FX_SMALL",
            "n_profiles": 3,
            "train_success_rate_mean": 0.99,
            "train_success_rate_std": 0.01,
            "train_mean_reward_mean": 0.0,
            "train_mean_reward_std": 0.0,
            "train_mean_steps_mean": 14.0,
            "train_mean_steps_std": 0.0,
            "train_mean_decisions_mean": 8.0,
            "train_mean_decisions_std": 0.0,
            "train_mean_action_duration_mean": 2.1,
            "train_mean_option_step_fraction_mean": 0.0,
            "train_first_success_episode_mean": 1.0,
            "train_first_success_episode_std": 0.0,
            "train_final_50_success_rate_mean": 1.0,
            "train_final_50_success_rate_std": 0.0,
            "eval_success_rate_mean": 1.0,
            "eval_success_rate_std": 0.0,
            "eval_mean_reward_mean": 0.0,
            "eval_mean_reward_std": 0.0,
            "eval_mean_steps_mean": 7.0,
            "eval_mean_steps_std": 0.0,
            "eval_mean_decisions_mean": 2.0,
            "eval_mean_decisions_std": 0.0,
            "eval_mean_action_duration_mean": 3.5,
            "eval_mean_option_step_fraction_mean": 0.0,
            "eval_mean_path_inefficiency_ratio_mean": 0.8,
            "eval_mean_path_inefficiency_ratio_std": 0.0,
            "eval_mean_wall_hits_mean": 0.0,
            "eval_mean_wall_hits_std": 0.0,
        },
    ]
    with (tmp_path / "summary_by_agent_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    (tmp_path / "manifest.json").write_text(
        json.dumps({"conditions_config": {"FX_SMALL": {"mode": "fixed", "width": 8, "height": 8}}}),
        encoding="utf-8",
    )

    outputs = generate_report_figures(str(tmp_path))

    assert len(outputs) == 4
    for output in outputs:
        path = Path(output)
        assert path.exists()
        assert path.parent == tmp_path / "figures"


def test_make_figures_resolves_latest_valid_results_dir(tmp_path: Path, monkeypatch) -> None:
    header = (
        "agent_type,condition,n_profiles,train_success_rate_mean,train_success_rate_std,"
        "train_mean_reward_mean,train_mean_reward_std,train_mean_steps_mean,train_mean_steps_std,"
        "train_mean_decisions_mean,train_mean_decisions_std,train_mean_action_duration_mean,"
        "train_mean_option_step_fraction_mean,train_first_success_episode_mean,"
        "train_first_success_episode_std,train_final_50_success_rate_mean,"
        "train_final_50_success_rate_std,eval_success_rate_mean,eval_success_rate_std,"
        "eval_mean_reward_mean,eval_mean_reward_std,eval_mean_steps_mean,eval_mean_steps_std,"
        "eval_mean_decisions_mean,eval_mean_decisions_std,eval_mean_action_duration_mean,"
        "eval_mean_option_step_fraction_mean,eval_mean_path_inefficiency_ratio_mean,"
        "eval_mean_path_inefficiency_ratio_std,eval_mean_wall_hits_mean,eval_mean_wall_hits_std\n"
    )
    row_template = (
        "{agent},FX_SMALL,3,0.9,0.0,0.0,0.0,10.0,0.0,10.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,"
        "0.0,0.0,7.0,0.0,7.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0\n"
    )
    valid_csv = header + "".join(
        row_template.format(agent=agent)
        for agent in ["DQN_LSTM_PRIMITIVE", "DQN_LSTM_NAIVE_MACROS", "DQN_LSTM_ASTAR_MACROS"]
    )

    results_root = tmp_path / "research" / "results"
    old_run = results_root / "old_run"
    new_run = results_root / "new_run"
    old_run.mkdir(parents=True)
    new_run.mkdir(parents=True)
    (old_run / "summary_by_agent_condition.csv").write_text(valid_csv, encoding="utf-8")
    (new_run / "summary_by_agent_condition.csv").write_text(valid_csv, encoding="utf-8")

    now = time.time()
    os.utime(old_run, (now - 10, now - 10))
    os.utime(new_run, (now, now))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.research.makeFigures.__file__", str(tmp_path / "code" / "services" / "research" / "makeFigures.py"))

    assert _resolve_experiment_dir(None) == new_run.resolve()
