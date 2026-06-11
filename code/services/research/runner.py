from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from typing import Any

from services.research.metrics import (
    coerce_episode_row,
    compute_profile_summary,
    episode_to_row,
)
from services.research.mazeExporter import (
    export_fixed_maze_json,
    export_pool_maze_json,
    load_fixed_maze,
    load_pool_mazes,
    load_pool_mazes_from_paths,
)
from services.research.plan import ConditionSpec, ExperimentPlan, ProfileSpec
from services.research.profileBuilder import build_bot, create_profile, delete_profile
from services.research.schema import normalize_export_keys
from services.research.writers import (
    refresh_observed_failure_metrics_from_episodes,
    write_break_even_tables_from_existing_exports,
    write_episodes_csv,
    write_final_report_tables,
    write_heatmap,
    write_json,
    write_plot_data,
    write_profile_diagnostics,
    write_profile_eval_csv,
    write_profile_summary,
    write_profile_training_csv,
    write_summary_by_agent_condition,
    write_summary_by_profile,
)

_PROGRESS_INTERVAL = 50


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _merge_heatmap(dest: dict[tuple[int, int], int], src: dict[tuple[int, int], int]) -> None:
    for pos, count in src.items():
        key = tuple(pos)
        dest[key] = dest.get(key, 0) + int(count)


def _serialize_heatmap(heatmap: dict[tuple[int, int], int]) -> dict[str, int]:
    return {str(tuple(pos)): int(count) for pos, count in heatmap.items()}


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _requirements_hash() -> str:
    for candidate in ("requirements.txt", "requirements-dev.txt"):
        if os.path.exists(candidate):
            with open(candidate, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
    return ""


def _gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "none"


def _ram_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        return None


def _cpu_info() -> str:
    return platform.processor() or "unknown"


def _episode_count_for_profile(spec: ProfileSpec) -> int:
    return int(spec.condition.training_episodes) + int(spec.condition.evaluation_episodes)


def _episode_count_for_curriculum(
    plan: ExperimentPlan,
    train_ids: list[str],
    eval_ids: list[str],
) -> int:
    return (
        sum(int(plan.conditions[cid].training_episodes) for cid in train_ids)
        + sum(int(plan.conditions[cid].evaluation_episodes) for cid in eval_ids)
    )


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------

class CheckpointStore:
    """Tracks which profiles have completed so an interrupted experiment can be resumed."""

    _FILENAME = "completed.json"

    def __init__(self, checkpoint_dir: str) -> None:
        self._path = os.path.join(checkpoint_dir, self._FILENAME)
        self._completed: set[str] = self._load()

    def _load(self) -> set[str]:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, TypeError, ValueError):
                return set()
        return set()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(sorted(self._completed), f, indent=2)

    def is_completed(self, profile_name: str) -> bool:
        return profile_name in self._completed

    def mark_completed(self, profile_name: str) -> None:
        self._completed.add(profile_name)
        self._save()

    def clear(self) -> None:
        self._completed.clear()
        if os.path.exists(self._path):
            os.remove(self._path)

    @property
    def count(self) -> int:
        return len(self._completed)


# ---------------------------------------------------------------------------
# MazeStateCache
# ---------------------------------------------------------------------------

class MazeStateCache:
    """Loads maze JSON files once and returns in-memory game states keyed by condition ID."""

    def __init__(self, plan_dir: str, mazes_out_dir: str) -> None:
        self._plan_dir = plan_dir
        self._mazes_out_dir = mazes_out_dir

    def prepare(self, profiles: list[ProfileSpec]) -> dict[str, Any]:
        seen: set[str] = set()
        result: dict[str, Any] = {}
        for spec in profiles:
            cid = spec.condition.condition_id
            if cid in seen:
                continue
            seen.add(cid)
            result[cid] = self._load_condition(spec.condition, cid)
        return result

    def _load_condition(self, cond: ConditionSpec, cid: str) -> Any:
        maze_out = os.path.join(self._mazes_out_dir, f"{cid}.json")
        if cond.mode == "fixed":
            maze, source_meta = load_fixed_maze(cond, self._plan_dir)
            write_json(
                maze_out,
                export_fixed_maze_json(
                    cond, maze,
                    source_path=str(source_meta["source_path"]),
                    source_sha256=str(source_meta["source_sha256"]),
                ),
            )
            return maze.getState()
        if cond.mode == "pool":
            return self._load_pool(cond, cid)
        return None

    def _load_pool(self, cond: ConditionSpec, cid: str) -> dict[str, list[Any]]:
        if cond.training_maze_paths and cond.evaluation_maze_paths:
            train_mazes, train_meta = load_pool_mazes_from_paths(cond, self._plan_dir, cond.training_maze_paths)
            eval_mazes, eval_meta = load_pool_mazes_from_paths(cond, self._plan_dir, cond.evaluation_maze_paths)
            write_json(
                os.path.join(self._mazes_out_dir, f"{cid}_train.json"),
                export_pool_maze_json(cond, train_mazes, train_meta, split="training"),
            )
            write_json(
                os.path.join(self._mazes_out_dir, f"{cid}_eval.json"),
                export_pool_maze_json(cond, eval_mazes, eval_meta, split="evaluation"),
            )
            return {
                "training": [m.getState() for m in train_mazes],
                "evaluation": [m.getState() for m in eval_mazes],
            }
        mazes, source_meta = load_pool_mazes(cond, self._plan_dir)
        write_json(
            os.path.join(self._mazes_out_dir, f"{cid}.json"),
            export_pool_maze_json(cond, mazes, source_meta),
        )
        states = [m.getState() for m in mazes]
        return {"training": states, "evaluation": states}


# ---------------------------------------------------------------------------
# ProfileExecutor
# ---------------------------------------------------------------------------

class ProfileExecutor:
    """Runs a single profile (bot lifecycle + episode loops) and writes all per-profile files."""

    def __init__(self, plan: ExperimentPlan, profile_store_dir: str) -> None:
        self._plan = plan
        self._profile_store_dir = profile_store_dir

    # ---- public entry points -----------------------------------------------

    def execute_standard(
        self,
        spec: ProfileSpec,
        maze_states: dict[str, Any],
        profile_out_dir: str,
        global_offset: int,
    ) -> None:
        os.makedirs(profile_out_dir, exist_ok=True)
        _set_all_seeds(spec.seed)

        maze, train_pool, eval_pool = self._build_maze(spec, maze_states)

        delete_profile(spec.profile_name, self._profile_store_dir)
        create_profile(spec, self._profile_store_dir, replay_warmup_policy=self._plan.dqn_replay_warmup_policy)
        bot = build_bot(spec, maze, self._profile_store_dir)

        train_rows, train_heatmaps = self._run_training(bot, spec, maze, train_pool, global_offset)
        self._persist_training_artifacts(bot)
        write_profile_training_csv(profile_out_dir, train_rows)

        eval_rows, eval_heatmaps = self._run_evaluation(
            bot, spec, maze, eval_pool, global_offset + len(train_rows)
        )
        write_profile_eval_csv(profile_out_dir, eval_rows)

        summary = compute_profile_summary(spec, train_rows, eval_rows, bot)
        write_profile_summary(profile_out_dir, summary)
        write_profile_diagnostics(profile_out_dir, self._build_diagnostics(bot, spec))

        write_heatmap(profile_out_dir, "training",
                      self._build_heatmap_payload(spec, "training", train_heatmaps, len(train_rows)))
        write_heatmap(profile_out_dir, "evaluation",
                      self._build_heatmap_payload(spec, "evaluation", eval_heatmaps, len(eval_rows)))

    def execute_curriculum(
        self,
        spec: ProfileSpec,
        train_ids: list[str],
        eval_ids: list[str],
        maze_states: dict[str, Any],
        profile_out_dir: str,
        global_offset: int,
    ) -> None:
        os.makedirs(profile_out_dir, exist_ok=True)
        _set_all_seeds(spec.seed)

        from maze import Maze
        initial_condition = self._plan.conditions[train_ids[0]]
        maze = Maze(initial_condition.width, initial_condition.height)
        initial_states = self._states_for_condition(initial_condition, maze_states, phase="training")
        if initial_states:
            maze.setState(initial_states[0])

        delete_profile(spec.profile_name, self._profile_store_dir)
        create_profile(spec, self._profile_store_dir, replay_warmup_policy=self._plan.dqn_replay_warmup_policy)
        bot = build_bot(spec, maze, self._profile_store_dir)

        all_train_rows: list[dict[str, Any]] = []
        all_eval_rows: list[dict[str, Any]] = []
        train_heatmaps: dict[str, dict[tuple[int, int], int]] = {}
        eval_heatmaps: dict[str, dict[tuple[int, int], int]] = {}
        cumulative_env_offset = 0
        cumulative_decision_offset = 0
        current_offset = global_offset

        for stage_num, condition_id in enumerate(train_ids, 1):
            stage_condition = self._plan.conditions[condition_id]
            stage_spec = self._stage_spec(spec, stage_condition)
            pool_states = self._states_for_condition(stage_condition, maze_states, phase="training")
            self._prepare_maze_for_stage(maze, stage_condition, pool_states, seed=spec.seed)
            print(
                f"  curriculum stage {stage_num}/{len(train_ids)} train: "
                f"{condition_id} ({stage_condition.training_episodes} episodes)"
            )
            stage_rows, stage_heatmaps = self._run_training(
                bot, stage_spec, maze,
                pool_states if stage_condition.mode == "pool" else [],
                current_offset,
            )
            self._offset_cumulative_rows(stage_rows, cumulative_env_offset, cumulative_decision_offset)
            if stage_rows:
                cumulative_env_offset = int(stage_rows[-1]["cumulative_env_steps"])
                cumulative_decision_offset = int(stage_rows[-1]["cumulative_decisions"])
            current_offset += len(stage_rows)
            all_train_rows.extend(stage_rows)
            self._merge_heatmaps_by_maze(train_heatmaps, stage_heatmaps)

        self._persist_training_artifacts(bot)
        write_profile_training_csv(profile_out_dir, all_train_rows)

        eval_env_offset = 0
        eval_decision_offset = 0
        for stage_num, condition_id in enumerate(eval_ids, 1):
            stage_condition = self._plan.conditions[condition_id]
            stage_spec = self._stage_spec(spec, stage_condition)
            pool_states = self._states_for_condition(stage_condition, maze_states, phase="evaluation")
            self._prepare_maze_for_stage(maze, stage_condition, pool_states, seed=spec.seed)
            print(
                f"  curriculum eval {stage_num}/{len(eval_ids)}: "
                f"{condition_id} ({stage_condition.evaluation_episodes} episodes)"
            )
            stage_rows, stage_heatmaps = self._run_evaluation(
                bot, stage_spec, maze,
                pool_states if stage_condition.mode == "pool" else [],
                current_offset,
            )
            self._offset_cumulative_rows(stage_rows, eval_env_offset, eval_decision_offset)
            if stage_rows:
                eval_env_offset = int(stage_rows[-1]["cumulative_env_steps"])
                eval_decision_offset = int(stage_rows[-1]["cumulative_decisions"])
            current_offset += len(stage_rows)
            all_eval_rows.extend(stage_rows)
            self._merge_heatmaps_by_maze(eval_heatmaps, stage_heatmaps)

        write_profile_eval_csv(profile_out_dir, all_eval_rows)

        summary = compute_profile_summary(spec, all_train_rows, all_eval_rows, bot)
        summary["curriculum"] = {"training_conditions": train_ids, "evaluation_conditions": eval_ids}
        write_profile_summary(profile_out_dir, summary)

        diagnostics = self._build_diagnostics(bot, spec)
        diagnostics["curriculum"] = summary["curriculum"]
        write_profile_diagnostics(profile_out_dir, diagnostics)

        write_heatmap(profile_out_dir, "training",
                      self._build_heatmap_payload(spec, "training", train_heatmaps, len(all_train_rows)))
        write_heatmap(profile_out_dir, "evaluation",
                      self._build_heatmap_payload(spec, "evaluation", eval_heatmaps, len(all_eval_rows)))

    # ---- episode loops -----------------------------------------------------

    def _run_training(
        self,
        bot: Any,
        spec: ProfileSpec,
        maze: Any,
        pool_states: list[Any],
        global_offset: int,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[tuple[int, int], int]]]:
        cond = spec.condition
        rows: list[dict[str, Any]] = []
        heatmaps_by_maze: dict[str, dict[tuple[int, int], int]] = {}
        rolling_rewards: deque[float] = deque(maxlen=_PROGRESS_INTERVAL)
        rolling_success: deque[float] = deque(maxlen=_PROGRESS_INTERVAL)
        cumulative_env_steps = 0
        cumulative_decisions = 0

        for ep in range(1, cond.training_episodes + 1):
            if pool_states:
                maze.setState(pool_states[(ep - 1) % len(pool_states)])

            maze_id = self._current_maze_id(spec, pool_states, (ep - 1) % max(1, len(pool_states)), split="train")
            t0 = time.monotonic()
            result = self._run_one_training_episode(bot)
            elapsed = time.monotonic() - t0

            row = episode_to_row(
                result=result, spec=spec, episode_num=ep, global_episode=global_offset + ep,
                phase="training", bot=bot, elapsed_seconds=elapsed,
                experiment_name=self._plan.experiment_name, maze_id=maze_id,
            )
            cumulative_env_steps += int(row["steps"])
            cumulative_decisions += int(row["decisions"])
            row["cumulative_env_steps"] = cumulative_env_steps
            row["cumulative_decisions"] = cumulative_decisions
            rows.append(row)

            _merge_heatmap(heatmaps_by_maze.setdefault(maze_id, {}), result.heatmap_data)
            rolling_rewards.append(result.total_reward)
            rolling_success.append(float(result.success))

            if ep % _PROGRESS_INTERVAL == 0:
                mean_r = sum(rolling_rewards) / len(rolling_rewards)
                mean_s = sum(rolling_success) / len(rolling_success)
                print(
                    f"  ep={ep}/{cond.training_episodes} train "
                    f"success_rolling={mean_s:.2f} reward_rolling={mean_r:.1f}"
                )

        return rows, heatmaps_by_maze

    def _run_evaluation(
        self,
        bot: Any,
        spec: ProfileSpec,
        maze: Any,
        pool_states: list[Any],
        global_offset: int,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[tuple[int, int], int]]]:
        cond = spec.condition
        rows: list[dict[str, Any]] = []
        heatmaps_by_maze: dict[str, dict[tuple[int, int], int]] = {}
        n_eval = cond.evaluation_episodes_per_pool_maze * len(pool_states) if pool_states else cond.evaluation_episodes
        cumulative_env_steps = 0
        cumulative_decisions = 0

        for ep in range(1, n_eval + 1):
            if pool_states:
                maze.setState(pool_states[(ep - 1) % len(pool_states)])

            maze_id = self._current_maze_id(spec, pool_states, (ep - 1) % max(1, len(pool_states)), split="eval")
            t0 = time.monotonic()
            result = self._run_one_eval_episode(bot)
            elapsed = time.monotonic() - t0

            row = episode_to_row(
                result=result, spec=spec, episode_num=ep, global_episode=global_offset + ep,
                phase="evaluation", bot=bot, elapsed_seconds=elapsed,
                experiment_name=self._plan.experiment_name, maze_id=maze_id,
            )
            cumulative_env_steps += int(row["steps"])
            cumulative_decisions += int(row["decisions"])
            row["cumulative_env_steps"] = cumulative_env_steps
            row["cumulative_decisions"] = cumulative_decisions
            rows.append(row)
            _merge_heatmap(heatmaps_by_maze.setdefault(maze_id, {}), result.heatmap_data)

        if rows:
            n = len(rows)
            print(
                f"  evaluation complete: "
                f"success={sum(r['success'] for r in rows) / n:.2f} "
                f"mean_reward={sum(r['total_reward'] for r in rows) / n:.1f} "
                f"mean_steps={sum(r['steps'] for r in rows) / n:.1f}"
            )

        return rows, heatmaps_by_maze

    # ---- episode invocation ------------------------------------------------

    @staticmethod
    def _run_one_training_episode(bot: Any) -> Any:
        runner = getattr(bot, "runner", None)
        if runner is None:
            raise RuntimeError(f"Bot {type(bot).__name__} has no 'runner' attribute")
        if not hasattr(runner, "runEpisode"):
            raise RuntimeError(f"Runner {type(runner).__name__} has no runEpisode method")
        return runner.runEpisode(mode="training")

    @staticmethod
    def _run_one_eval_episode(bot: Any) -> Any:
        runner = getattr(bot, "runner", None)
        if runner is None:
            raise RuntimeError(f"Bot {type(bot).__name__} has no 'runner' attribute")
        if hasattr(runner, "runEvaluationEpisode"):
            return runner.runEvaluationEpisode()
        if hasattr(runner, "runEpisode"):
            return runner.runEpisode(mode="evaluation")
        raise RuntimeError(f"Runner {type(runner).__name__} has no evaluation episode method")

    # ---- helpers -----------------------------------------------------------

    def _build_maze(
        self,
        spec: ProfileSpec,
        maze_states: dict[str, Any],
    ) -> tuple[Any, list[Any], list[Any]]:
        from maze import Maze
        cond = spec.condition
        maze_state = maze_states.get(cond.condition_id)
        train_pool: list[Any] = []
        eval_pool: list[Any] = []

        if cond.mode == "fixed" and maze_state is not None:
            maze = Maze(cond.width, cond.height)
            maze.setState(maze_state)
        elif cond.mode == "pool" and isinstance(maze_state, dict):
            train_pool = list(maze_state.get("training", []))
            eval_pool = list(maze_state.get("evaluation", []))
            maze = Maze(cond.width, cond.height)
            maze.setState((train_pool or eval_pool)[0])
        else:
            # random mode — seed already set by caller; generate a pool so episodes
            # see varied mazes rather than repeating the same one every episode
            pool_size = min(cond.training_episodes, 20)
            train_pool = [Maze(cond.width, cond.height).getState() for _ in range(pool_size)]
            maze = Maze(cond.width, cond.height)
            maze.setState(train_pool[0])

        return maze, train_pool, eval_pool

    @staticmethod
    def _persist_training_artifacts(bot: Any) -> None:
        persist = getattr(bot, "persistTrainingArtifacts", None)
        if callable(persist):
            persist()

    @staticmethod
    def _build_diagnostics(bot: Any, spec: ProfileSpec) -> dict[str, Any]:
        diag: dict[str, Any] = {
            "profile_name": spec.profile_name,
            "agent_type": spec.agent_type,
            "condition": spec.condition.condition_id,
            "seed": spec.seed,
        }
        q_learner = getattr(bot, "qLearning", None)
        if q_learner is not None:
            diag["q_table_states"] = len(q_learner.qTable)
            diag["final_epsilon"] = round(q_learner.explorationRate(), 6)
            diag["total_steps"] = q_learner.totalSteps
        agent = getattr(bot, "agent", None)
        if agent is not None:
            diag_fn = getattr(agent, "diagnostics", None)
            if callable(diag_fn):
                agent_diag = diag_fn()
                diag["final_replay_size"] = getattr(agent_diag, "replaySize", None)
                diag["final_epsilon"] = getattr(agent_diag, "epsilon", None)
                diag["final_training_updates"] = getattr(agent_diag, "trainingUpdates", None)
                diag["final_loss"] = getattr(agent_diag, "lastLoss", None)
                diag["warmup_completion_step"] = getattr(agent_diag, "warmupCompletionStep", None)
            else:
                replay = getattr(agent, "replay", None) or getattr(agent, "_replay", None)
                if replay is not None:
                    diag["final_replay_size"] = len(replay)
                epsilon_fn = getattr(agent, "_epsilon", None)
                diag["final_epsilon"] = epsilon_fn() if callable(epsilon_fn) else getattr(agent, "epsilon", None)
                diag["final_training_updates"] = getattr(agent, "trainingUpdates", None)
                diag["final_loss"] = getattr(agent, "lastLoss", None)
        option_diag_fn = getattr(bot, "optionDiagnostics", None)
        if callable(option_diag_fn):
            diag["option_diagnostics"] = option_diag_fn()
        return diag

    @staticmethod
    def _build_heatmap_payload(
        spec: ProfileSpec,
        phase: str,
        heatmaps_by_maze: dict[str, dict[tuple[int, int], int]],
        episodes_aggregated: int,
    ) -> dict[str, Any]:
        aggregate: dict[tuple[int, int], int] = {}
        for maze_heatmap in heatmaps_by_maze.values():
            _merge_heatmap(aggregate, maze_heatmap)
        maze_ids = sorted(heatmaps_by_maze.keys())
        return {
            "profile_name": spec.profile_name,
            "condition": spec.condition.condition_id,
            "phase": phase,
            "width": spec.condition.width,
            "height": spec.condition.height,
            "episodes_aggregated": episodes_aggregated,
            "aggregation": "sum_over_episodes",
            "maze_scope": "single" if len(maze_ids) <= 1 else "multiple",
            "maze_id": maze_ids[0] if len(maze_ids) == 1 else None,
            "maze_ids": maze_ids,
            "heatmap": _serialize_heatmap(aggregate),
            "heatmaps_by_maze": {
                maze_id: _serialize_heatmap(hm)
                for maze_id, hm in sorted(heatmaps_by_maze.items())
            },
        }

    @staticmethod
    def _current_maze_id(spec: ProfileSpec, pool_states: list[Any], pool_idx: int, *, split: str) -> str:
        cond = spec.condition
        if cond.mode == "fixed":
            return cond.maze_id
        if cond.mode == "pool":
            if cond.training_maze_paths and cond.evaluation_maze_paths:
                return f"{cond.maze_id_prefix}_{split}_{pool_idx:02d}"
            return f"{cond.maze_id_prefix}_{pool_idx:02d}"
        return "random"

    @staticmethod
    def _stage_spec(base: ProfileSpec, condition: ConditionSpec) -> ProfileSpec:
        return ProfileSpec(
            profile_name=base.profile_name,
            agent_type=base.agent_type,
            bot_type=base.bot_type,
            condition=condition,
            seed_label=base.seed_label,
            seed=base.seed,
        )

    @staticmethod
    def _states_for_condition(
        condition: ConditionSpec,
        maze_states: dict[str, Any],
        *,
        phase: str,
    ) -> list[Any]:
        state = maze_states.get(condition.condition_id)
        if condition.mode == "pool" and isinstance(state, dict):
            key = "evaluation" if phase == "evaluation" else "training"
            return list(state.get(key, []))
        if state is not None and condition.mode == "fixed":
            return [state]
        return []

    @staticmethod
    def _prepare_maze_for_stage(maze: Any, condition: ConditionSpec, states: list[Any], *, seed: int = 0) -> None:
        if condition.mode in {"fixed", "pool"} and states:
            maze.setState(states[0])
        elif condition.mode == "random":
            _set_all_seeds(seed)
            from maze import Maze
            replacement = Maze(condition.width, condition.height)
            maze.setState(replacement.getState())

    @staticmethod
    def _offset_cumulative_rows(
        rows: list[dict[str, Any]],
        env_offset: int,
        decision_offset: int,
    ) -> None:
        if not env_offset and not decision_offset:
            return
        for row in rows:
            row["cumulative_env_steps"] = int(row.get("cumulative_env_steps", 0)) + env_offset
            row["cumulative_decisions"] = int(row.get("cumulative_decisions", 0)) + decision_offset

    @staticmethod
    def _merge_heatmaps_by_maze(
        dest: dict[str, dict[tuple[int, int], int]],
        src: dict[str, dict[tuple[int, int], int]],
    ) -> None:
        for maze_id, heatmap in src.items():
            _merge_heatmap(dest.setdefault(maze_id, {}), heatmap)


# ---------------------------------------------------------------------------
# ResearchRunner
# ---------------------------------------------------------------------------

class ResearchRunner:
    """Coordinates a full experiment: maze loading, profile execution, and output aggregation."""

    def __init__(
        self,
        plan: ExperimentPlan,
        plan_path: str,
        out_dir: str,
    ) -> None:
        self.plan = plan
        self.plan_path = plan_path
        self.out_dir = out_dir
        self.profiles_out_dir = os.path.join(out_dir, "profiles")
        self.mazes_out_dir = os.path.join(out_dir, "mazes")
        self.errors_out_dir = os.path.join(out_dir, "errors")
        self.profile_store_dir = os.path.join(out_dir, "_profile_store")

    def run(self, *, resume: bool = False) -> None:
        if self.plan.raw.get("curriculum"):
            self._run_curriculum(resume=resume)
            return

        if not resume:
            shutil.rmtree(self.out_dir, ignore_errors=True)

        profiles = self.plan.expand_profiles()
        n_train = sum(p.condition.training_episodes for p in profiles)
        n_eval = sum(p.condition.evaluation_episodes for p in profiles)

        print(f"[Research] Plan: {self.plan.experiment_name}")
        print(f"[Research] Output: {self.out_dir}")
        print(f"[Research] Profiles planned: {len(profiles)}")
        print(f"[Research] Training episodes planned: {n_train}")
        print(f"[Research] Evaluation episodes planned: {n_eval}")
        print(f"[Research] DQN replay warmup policy: {self.plan.dqn_replay_warmup_policy}")

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.profiles_out_dir, exist_ok=True)
        os.makedirs(self.mazes_out_dir, exist_ok=True)
        os.makedirs(self.errors_out_dir, exist_ok=True)
        os.makedirs(self.profile_store_dir, exist_ok=True)

        checkpoint = CheckpointStore(self.profile_store_dir)
        if not resume:
            checkpoint.clear()
        elif checkpoint.count > 0:
            print(f"[Research] Resuming: {checkpoint.count} profile(s) already complete")

        manifest = self._build_manifest(profiles)
        if resume:
            manifest = self._merge_resume_manifest(manifest)
        write_json(os.path.join(self.out_dir, "manifest.json"), manifest)

        plan_dir = os.path.dirname(self.plan_path)
        maze_states = MazeStateCache(plan_dir, self.mazes_out_dir).prepare(profiles)
        executor = ProfileExecutor(self.plan, self.profile_store_dir)

        completed, failed, all_episode_rows, all_summaries = self._execute_profile_batch(
            profiles=profiles,
            maze_states=maze_states,
            executor=executor,
            checkpoint=checkpoint,
            mode="standard",
        )

        write_episodes_csv(os.path.join(self.out_dir, "episodes.csv"), all_episode_rows)
        write_summary_by_profile(self.out_dir, all_summaries)
        write_summary_by_agent_condition(self.out_dir, all_summaries)
        write_plot_data(self.out_dir, all_episode_rows)
        refresh_observed_failure_metrics_from_episodes(self.out_dir)
        write_break_even_tables_from_existing_exports(self.out_dir)
        write_final_report_tables(self.out_dir, self.plan.raw, all_summaries)

        manifest["completed_profiles"] = completed
        manifest["failed_profiles"] = failed
        write_json(os.path.join(self.out_dir, "manifest.json"), manifest)

        self._generate_figures()

        print("\n[Research] Complete")
        print(f"  completed_profiles: {len(completed)}")
        print(f"  failed_profiles: {len(failed)}")
        print("  episodes.csv saved")
        print("  summary tables saved")
        print("  manifest saved")

    def _run_curriculum(self, *, resume: bool) -> None:
        curriculum = self.plan.raw.get("curriculum", {})
        train_ids = [str(c) for c in curriculum.get("training_conditions", [])]
        eval_ids = [str(c) for c in curriculum.get("evaluation_conditions", [])]
        if not train_ids:
            raise ValueError("Curriculum plan requires curriculum.training_conditions")
        if not eval_ids:
            raise ValueError("Curriculum plan requires curriculum.evaluation_conditions")
        for cid in train_ids + eval_ids:
            if cid not in self.plan.conditions:
                raise ValueError(f"Curriculum references unknown condition '{cid}'")

        final_condition = self.plan.conditions[train_ids[-1]]
        curriculum_condition = ConditionSpec(
            condition_id=str(curriculum.get("condition_id", "CURRICULUM")),
            mode="curriculum",
            width=final_condition.width,
            height=final_condition.height,
            training_episodes=sum(self.plan.conditions[cid].training_episodes for cid in train_ids),
            evaluation_episodes=sum(self.plan.conditions[cid].evaluation_episodes for cid in eval_ids),
            maze_id=str(curriculum.get("condition_id", "curriculum")),
        )
        profiles: list[ProfileSpec] = []
        for agent in self.plan.agents:
            bot_type = self.plan.agent_to_bot_type(agent)
            for seed_label, seed in zip(self.plan.seed_labels, self.plan.seeds):
                profiles.append(ProfileSpec(
                    profile_name=f"{agent}_{curriculum_condition.condition_id}_{seed_label}",
                    agent_type=agent,
                    bot_type=bot_type,
                    condition=curriculum_condition,
                    seed_label=seed_label,
                    seed=seed,
                ))

        n_train = sum(self.plan.conditions[cid].training_episodes for cid in train_ids) * len(profiles)
        n_eval = sum(self.plan.conditions[cid].evaluation_episodes for cid in eval_ids) * len(profiles)

        print(f"[Research] Plan: {self.plan.experiment_name}")
        print(f"[Research] Curriculum: {' -> '.join(train_ids)}; eval={','.join(eval_ids)}")
        print(f"[Research] Output: {self.out_dir}")
        print(f"[Research] Profiles planned: {len(profiles)}")
        print(f"[Research] Training episodes planned: {n_train}")
        print(f"[Research] Evaluation episodes planned: {n_eval}")
        print(f"[Research] DQN replay warmup policy: {self.plan.dqn_replay_warmup_policy}")

        if not resume:
            shutil.rmtree(self.out_dir, ignore_errors=True)

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.profiles_out_dir, exist_ok=True)
        os.makedirs(self.mazes_out_dir, exist_ok=True)
        os.makedirs(self.errors_out_dir, exist_ok=True)
        os.makedirs(self.profile_store_dir, exist_ok=True)

        checkpoint = CheckpointStore(self.profile_store_dir)
        if not resume:
            checkpoint.clear()
        elif checkpoint.count > 0:
            print(f"[Research] Resuming: {checkpoint.count} profile(s) already complete")

        condition_specs = [
            ProfileSpec(
                profile_name="maze_export",
                agent_type=profiles[0].agent_type if profiles else "DQN_LSTM_PRIMITIVE",
                bot_type=profiles[0].bot_type if profiles else "DQNBot",
                condition=self.plan.conditions[cid],
                seed_label="S0",
                seed=self.plan.seeds[0],
            )
            for cid in list(dict.fromkeys(train_ids + eval_ids))
        ]
        plan_dir = os.path.dirname(self.plan_path)
        maze_states = MazeStateCache(plan_dir, self.mazes_out_dir).prepare(condition_specs)

        manifest = self._build_curriculum_manifest(profiles, train_ids, eval_ids, n_train, n_eval)
        if resume:
            manifest = self._merge_resume_manifest(manifest)
        write_json(os.path.join(self.out_dir, "manifest.json"), manifest)

        executor = ProfileExecutor(self.plan, self.profile_store_dir)
        completed, failed, all_episode_rows, all_summaries = self._execute_profile_batch(
            profiles=profiles,
            maze_states=maze_states,
            executor=executor,
            checkpoint=checkpoint,
            mode="curriculum",
            train_ids=train_ids,
            eval_ids=eval_ids,
        )

        write_episodes_csv(os.path.join(self.out_dir, "episodes.csv"), all_episode_rows)
        write_summary_by_profile(self.out_dir, all_summaries)
        write_summary_by_agent_condition(self.out_dir, all_summaries)
        write_plot_data(self.out_dir, all_episode_rows)
        refresh_observed_failure_metrics_from_episodes(self.out_dir)
        write_break_even_tables_from_existing_exports(self.out_dir)
        write_final_report_tables(self.out_dir, self.plan.raw, all_summaries)
        manifest["completed_profiles"] = completed
        manifest["failed_profiles"] = failed
        write_json(os.path.join(self.out_dir, "manifest.json"), manifest)

        self._generate_figures()

        print("\n[Research] Complete")
        print(f"  completed_profiles: {len(completed)}")
        print(f"  failed_profiles: {len(failed)}")

    def _execute_profile_batch(
        self,
        *,
        profiles: list[ProfileSpec],
        maze_states: dict[str, Any],
        executor: ProfileExecutor,
        checkpoint: CheckpointStore,
        mode: str,
        train_ids: list[str] | None = None,
        eval_ids: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        completed: list[str] = []
        failed: list[str] = []
        profile_rows: list[tuple[int, str]] = []
        profile_summaries: list[tuple[int, dict[str, Any]]] = []
        global_episode = 0

        for i, spec in enumerate(profiles, 1):
            profile_out_dir = os.path.join(self.profiles_out_dir, spec.profile_name)
            episode_span = (
                _episode_count_for_curriculum(self.plan, train_ids or [], eval_ids or [])
                if mode == "curriculum"
                else _episode_count_for_profile(spec)
            )

            if checkpoint.is_completed(spec.profile_name):
                print(f"\n[Profile {i:02d}/{len(profiles):02d}] {spec.profile_name}")
                print("  status: skipped (already complete, loading from disk)")
                try:
                    summary = self._load_summary(profile_out_dir)
                    profile_rows.append((i, profile_out_dir))
                    profile_summaries.append((i, summary))
                    completed.append(spec.profile_name)
                except Exception as exc:
                    print(f"  [WARNING] Could not reload {spec.profile_name}: {exc}")
                global_episode += episode_span
                continue

            if mode == "curriculum":
                print(f"\n[Profile {i:02d}/{len(profiles):02d}] {spec.profile_name}")
                print(f"  curriculum training: {' -> '.join(train_ids or [])}")
                print(f"  curriculum evaluation: {', '.join(eval_ids or [])}")
            else:
                cond = spec.condition
                print(f"\n[Profile {i:02d}/{len(profiles):02d}] {spec.profile_name}")
                print(f"  condition: {cond.condition_id} {cond.mode} {cond.width}x{cond.height}")
                print(f"  training: {cond.training_episodes} episodes")
                print(f"  evaluation: {cond.evaluation_episodes} episodes")
            print("  status: running")

            try:
                if mode == "curriculum":
                    executor.execute_curriculum(
                        spec=spec,
                        train_ids=list(train_ids or []),
                        eval_ids=list(eval_ids or []),
                        maze_states=maze_states,
                        profile_out_dir=profile_out_dir,
                        global_offset=global_episode,
                    )
                else:
                    executor.execute_standard(
                        spec=spec,
                        maze_states=maze_states,
                        profile_out_dir=profile_out_dir,
                        global_offset=global_episode,
                    )
                checkpoint.mark_completed(spec.profile_name)
                completed.append(spec.profile_name)
                profile_rows.append((i, profile_out_dir))
                profile_summaries.append((i, self._load_summary(profile_out_dir)))
                print(f"  saved: {os.path.join(profile_out_dir, 'summary.json')}")
            except Exception as exc:
                tb = traceback.format_exc()
                failed.append(spec.profile_name)
                self._write_failure_log(spec, exc, tb)
                print(f"  [ERROR] {spec.profile_name} failed: {exc}")

            global_episode += episode_span

        ordered_rows: list[dict[str, Any]] = []
        for _, profile_out_dir in sorted(profile_rows, key=lambda item: item[0]):
            ordered_rows.extend(self._load_profile_episode_rows(profile_out_dir))
        ordered_summaries = [s for _, s in sorted(profile_summaries, key=lambda item: item[0])]
        return completed, failed, ordered_rows, ordered_summaries

    def _generate_figures(self) -> None:
        try:
            from services.research.plotting import generate_figures
            figures = generate_figures(self.out_dir, os.path.join(self.out_dir, "plot_data"))
            if figures:
                print(f"\n[Research] Generated {len(figures)} figure(s)")
        except Exception as exc:
            print(f"[Research] Figure generation failed (non-fatal): {exc}")

        try:
            from services.research.makeFigures import generate_report_figures
            report_figures = generate_report_figures(self.out_dir)
            if report_figures:
                print(f"[Research] Generated {len(report_figures)} report figure(s)")
        except Exception as exc:
            print(f"[Research] Report figure generation failed (non-fatal): {exc}")

    def _merge_resume_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Preserve original created_at when resuming; add resumed_at timestamp."""
        existing_path = os.path.join(self.out_dir, "manifest.json")
        if os.path.exists(existing_path):
            try:
                with open(existing_path, "r", encoding="utf-8") as f:
                    existing = normalize_export_keys(json.load(f))
                manifest["created_at"] = existing.get("created_at", manifest["created_at"])
            except (json.JSONDecodeError, OSError):
                pass
        manifest["resumed_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        return manifest

    def _capture_environment(self) -> dict[str, Any]:
        return {
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "branch": self.plan.branch,
            "commit_hash": _git_commit_hash(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "cpu": _cpu_info(),
            "gpu": _gpu_info(),
            "ram_gb": _ram_gb(),
            "requirements_hash": _requirements_hash(),
        }

    def _build_manifest(self, profiles: list[ProfileSpec]) -> dict[str, Any]:
        agents = list(dict.fromkeys(p.agent_type for p in profiles))
        conditions = list(dict.fromkeys(p.condition.condition_id for p in profiles))
        n_train = sum(p.condition.training_episodes for p in profiles)
        n_eval = sum(p.condition.evaluation_episodes for p in profiles)
        return {
            "experiment_name": self.plan.experiment_name,
            **self._capture_environment(),
            "plan_path": self.plan_path,
            "output_dir": self.out_dir,
            "agents": agents,
            "conditions": conditions,
            "conditions_config": self.plan.raw.get("conditions", {}),
            "seeds": self.plan.seeds,
            "dqn_replay_warmup_policy": self.plan.dqn_replay_warmup_policy,
            "total_profiles": len(profiles),
            "total_training_episodes_planned": n_train,
            "total_evaluation_episodes_planned": n_eval,
            "completed_profiles": [],
            "failed_profiles": [],
        }

    def _build_curriculum_manifest(
        self,
        profiles: list[ProfileSpec],
        train_ids: list[str],
        eval_ids: list[str],
        n_train: int,
        n_eval: int,
    ) -> dict[str, Any]:
        agents = list(dict.fromkeys(p.agent_type for p in profiles))
        curriculum = dict(self.plan.raw.get("curriculum", {}))
        return {
            "experiment_name": self.plan.experiment_name,
            **self._capture_environment(),
            "plan_path": self.plan_path,
            "output_dir": self.out_dir,
            "agents": agents,
            "conditions": list(dict.fromkeys(train_ids + eval_ids)),
            "conditions_config": self.plan.raw.get("conditions", {}),
            "curriculum": curriculum,
            "seeds": self.plan.seeds,
            "dqn_replay_warmup_policy": self.plan.dqn_replay_warmup_policy,
            "total_profiles": len(profiles),
            "total_training_episodes_planned": n_train,
            "total_evaluation_episodes_planned": n_eval,
            "completed_profiles": [],
            "failed_profiles": [],
        }

    def _write_failure_log(self, spec: ProfileSpec, exc: Exception, tb: str) -> None:
        os.makedirs(self.errors_out_dir, exist_ok=True)
        write_json(
            os.path.join(self.errors_out_dir, f"{spec.profile_name}.log"),
            {
                "profile_name": spec.profile_name,
                "agent_type": spec.agent_type,
                "condition": spec.condition.condition_id,
                "run_id": spec.seed_label,
                "seed": spec.seed,
                "phase": "unknown",
                "episode": None,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb,
            },
        )

    def _load_profile_episode_rows(self, profile_out_dir: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in ("training_episodes.csv", "evaluation_episodes.csv"):
            path = os.path.join(profile_out_dir, name)
            if not os.path.exists(path):
                continue
            with open(path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append(coerce_episode_row(row))
        return rows

    def _load_summary(self, profile_out_dir: str) -> dict[str, Any]:
        with open(os.path.join(profile_out_dir, "summary.json"), "r", encoding="utf-8") as f:
            return dict(normalize_export_keys(json.load(f)))
