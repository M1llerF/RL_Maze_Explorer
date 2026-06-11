from __future__ import annotations

import contextlib
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.research.plan import ConditionSpec, ExperimentPlan
from services.research.runner import ResearchRunner


REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_RESULTS_ROOT = REPO_ROOT / "research" / "results"
RESEARCH_FIGURES_ROOT = REPO_ROOT / "research" / "figures"


@dataclass(frozen=True)
class ResearchPreset:
    preset_id: str
    label: str
    description: str
    plan_path: Path
    output_dir: Path


@dataclass(frozen=True)
class ResearchPlanSummary:
    preset: ResearchPreset
    experiment_name: str
    branch: str
    agents: tuple[str, ...]
    seeds: tuple[int, ...]
    seed_labels: tuple[str, ...]
    condition_lines: tuple[str, ...]
    curriculum_training: tuple[str, ...]
    curriculum_evaluation: tuple[str, ...]
    total_profiles: int
    total_training_episodes: int
    total_evaluation_episodes: int


def get_static_research_presets(repo_root: Path | None = None) -> tuple[ResearchPreset, ...]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    plans_root = root / "research" / "plans"
    results_root = root / "research" / "results"
    return (
        ResearchPreset(
            preset_id="macro_action_quality",
            label="Macro Action Quality",
            description=(
                "Supported paper benchmark: primitive vs naive, momentum, and A* macro agents on fixed and held-out pool mazes."
            ),
            plan_path=plans_root / "lstm_macro_plan.json",
            output_dir=results_root / "lstm_macro_run",
        ),
    )


def get_research_preset(preset_id: str, repo_root: Path | None = None) -> ResearchPreset:
    for preset in get_static_research_presets(repo_root):
        if preset.preset_id == preset_id:
            return preset
    raise KeyError(f"Unknown research preset '{preset_id}'")


def load_research_summary(preset: ResearchPreset) -> ResearchPlanSummary:
    plan = ExperimentPlan.load(os.fspath(preset.plan_path))
    curriculum_raw = dict(plan.raw.get("curriculum", {}))
    curriculum_training = tuple(str(cid) for cid in curriculum_raw.get("training_conditions", []))
    curriculum_evaluation = tuple(str(cid) for cid in curriculum_raw.get("evaluation_conditions", []))

    if curriculum_training or curriculum_evaluation:
        total_profiles = len(plan.agents) * len(plan.seeds)
        total_training_episodes = (
            sum(plan.conditions[cid].training_episodes for cid in curriculum_training) * total_profiles
        )
        total_evaluation_episodes = (
            sum(plan.conditions[cid].evaluation_episodes for cid in curriculum_evaluation) * total_profiles
        )
    else:
        total_profiles = len(plan.expand_profiles())
        total_training_episodes = plan.total_training_episodes()
        total_evaluation_episodes = plan.total_evaluation_episodes()

    condition_lines = tuple(_format_condition_line(plan.conditions[cid]) for cid in plan.conditions)
    return ResearchPlanSummary(
        preset=preset,
        experiment_name=plan.experiment_name,
        branch=plan.branch,
        agents=tuple(plan.agents),
        seeds=tuple(plan.seeds),
        seed_labels=tuple(plan.seed_labels),
        condition_lines=condition_lines,
        curriculum_training=curriculum_training,
        curriculum_evaluation=curriculum_evaluation,
        total_profiles=total_profiles,
        total_training_episodes=total_training_episodes,
        total_evaluation_episodes=total_evaluation_episodes,
    )


def load_research_runner(preset: ResearchPreset, output_dir: str | None = None) -> ResearchRunner:
    plan = ExperimentPlan.load(os.fspath(preset.plan_path))
    effective_out = output_dir if output_dir is not None else os.fspath(preset.output_dir)
    return ResearchRunner(
        plan=plan,
        plan_path=os.fspath(preset.plan_path),
        out_dir=effective_out,
    )


def run_research_preset(preset_id: str, *, resume: bool = False, output_dir: str | None = None) -> None:
    preset = get_research_preset(preset_id)
    runner = load_research_runner(preset, output_dir)
    runner.run(resume=resume)


def start_research_process(
    preset_id: str, *, resume: bool = False, output_dir: str | None = None
) -> tuple[Any, Any]:
    context = multiprocessing.get_context("spawn")
    log_queue = context.Queue()
    process = context.Process(
        target=_run_research_process,
        args=(preset_id, log_queue, resume, output_dir),
        daemon=True,
    )
    process.start()
    return process, log_queue


def format_research_summary(summary: ResearchPlanSummary, repo_root: Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    output_dir = _relative_or_absolute(summary.preset.output_dir, root)
    lines = [
        f"Experiment Label: {summary.preset.label}",
        f"Description: {summary.preset.description}",
        f"Experiment: {summary.experiment_name}",
        f"Branch: {summary.branch}",
        f"Plan: {_relative_or_absolute(summary.preset.plan_path, root)}",
        f"Output: {output_dir}",
        "Run mode: GUI only; starting a run replaces this output folder.",
        f"Agents: {', '.join(summary.agents)}",
        f"Seeds: {', '.join(f'{label}={seed}' for label, seed in zip(summary.seed_labels, summary.seeds))}",
        f"Profiles: {summary.total_profiles}",
        f"Training episodes: {summary.total_training_episodes}",
        f"Evaluation episodes: {summary.total_evaluation_episodes}",
    ]
    if summary.curriculum_training or summary.curriculum_evaluation:
        lines.append(f"Curriculum train: {' -> '.join(summary.curriculum_training)}")
        lines.append(f"Curriculum eval: {', '.join(summary.curriculum_evaluation)}")
    lines.append("Conditions:")
    lines.extend(f"  {line}" for line in summary.condition_lines)
    return "\n".join(lines)


class _QueueWriter:
    def __init__(self, log_queue: Any) -> None:
        self._log_queue = log_queue
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._log_queue.put(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._log_queue.put(self._buffer)
            self._buffer = ""


def _run_research_process(
    preset_id: str, log_queue: Any, resume: bool = False, output_dir: str | None = None
) -> None:
    writer = _QueueWriter(log_queue)
    with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
        preset = get_research_preset(preset_id)
        effective_out = output_dir if output_dir is not None else os.fspath(preset.output_dir)
        print(f"Launching experiment '{preset.label}'")
        print(f"Plan: {preset.plan_path}")
        print(f"Output: {effective_out}")
        if resume:
            print("Resuming from previous run.")
        else:
            print("Starting fresh — existing output will be replaced.")
        run_research_preset(preset_id, resume=resume, output_dir=output_dir)
        writer.flush()


def _format_condition_line(condition: ConditionSpec) -> str:
    if condition.mode == "fixed":
        source = Path(condition.maze_path).name or "maze file"
    elif condition.mode == "pool":
        if condition.training_maze_paths and condition.evaluation_maze_paths:
            source = (
                f"{len(condition.training_maze_paths)} train mazes "
                f"/ {len(condition.evaluation_maze_paths)} eval mazes"
            )
        else:
            source = f"{len(condition.maze_paths)} shared mazes"
    else:
        source = condition.mode

    return (
        f"{condition.condition_id} | {condition.mode} {condition.width}x{condition.height} | "
        f"train {condition.training_episodes} | eval {condition.evaluation_episodes} | {source}"
    )


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return os.fspath(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return os.fspath(path.resolve())
