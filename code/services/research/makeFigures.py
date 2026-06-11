#!/usr/bin/env python3
"""
Generate the four report figures for a research output directory.

By default the script reads the current working directory if it already looks
like a research output directory. Otherwise it falls back to the newest valid
directory under repo_root/research/results.

Usage:
    python code/services/research/makeFigures.py
    python code/services/research/makeFigures.py research/results/lstm_macro_run

The figures are written to repo_root/research/figures by default.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

CODE_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(CODE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(CODE_ROOT))

from services.research.schema import normalize_export_keys

ReportMetrics = dict[str, float]
ConditionReportData = dict[str, ReportMetrics]
ReportData = dict[str, ConditionReportData]
FigureRenderer = Callable[["PlotContext", Path], None]

DEFAULT_CONDS: Final[tuple[str, ...]] = (
    "FX_SMALL",
    "FX_MED",
    "FX_LARGE_25",
    "POOL_MED_HELDOUT",
)
AGENTS: Final[tuple[str, ...]] = (
    "DQN_LSTM_PRIMITIVE",
    "DQN_LSTM_NAIVE_MACROS",
    "DQN_LSTM_MOMENTUM_MACROS",
    "DQN_LSTM_ASTAR_MACROS",
)
MACRO_AGENTS: Final[tuple[str, ...]] = tuple(
    agent for agent in AGENTS if agent != "DQN_LSTM_PRIMITIVE"
)
AGENT_LABELS: Final[dict[str, str]] = {
    "DQN_LSTM_PRIMITIVE": "Primitive",
    "DQN_LSTM_NAIVE_MACROS": "Naive macros",
    "DQN_LSTM_MOMENTUM_MACROS": "Momentum macros",
    "DQN_LSTM_ASTAR_MACROS": "A* macros (oracle)",
}
AGENT_COLORS: Final[dict[str, str]] = {
    "DQN_LSTM_PRIMITIVE": "#7f7f7f",
    "DQN_LSTM_NAIVE_MACROS": "#e8843c",
    "DQN_LSTM_MOMENTUM_MACROS": "#4c9a65",
    "DQN_LSTM_ASTAR_MACROS": "#2a7ab0",
}
CONDITION_MARKERS: Final[dict[str, str]] = {
    "FX_SMALL": "o",
    "FX_MED": "s",
    "FX_LARGE_25": "^",
    "POOL_MED_HELDOUT": "D",
}
FALLBACK_CONDITION_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
LOG_MINOR_SUBS: Final[tuple[float, ...]] = tuple(step / 10.0 for step in range(2, 10))
DEFAULT_LAMBDA_FAILURE: Final[float] = 1.0
LEGACY_FIGURE_NAMES: Final[tuple[str, ...]] = ("fig3_breakeven.png", "fig4_tau.png")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


@dataclass(frozen=True)
class LoadedReportData:
    data: ReportData
    manifest: dict[str, Any]
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class PlotContext:
    data: ReportData
    conditions: tuple[str, ...]
    condition_labels: dict[str, str]


def _new_figure(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return cast(Figure, fig), cast(Axes, ax)


def _save_figure(fig: Figure, out_path: Path) -> None:
    fig.savefig(out_path)
    plt.close(fig)


def _apply_common_axes_style(ax: Axes) -> None:
    ax.grid(True, which="major", alpha=0.2, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.4)
    ax.tick_params(axis="both", which="major", length=4)
    ax.tick_params(axis="both", which="minor", length=2.5)


def _configure_log_x_axis(ax: Axes) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=LOG_MINOR_SUBS, numticks=100)
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def _callout_box() -> dict[str, Any]:
    return {
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.84,
    }


def _legend_box_style() -> dict[str, Any]:
    return {"frameon": True, "framealpha": 0.94, "edgecolor": "#bbbbbb"}


def _condition_label(condition_id: str, manifest: dict[str, Any]) -> str:
    raw = dict(manifest.get("conditions_config", {})).get(condition_id, {})
    width = raw.get("width")
    height = raw.get("height")
    mode = raw.get("mode")
    if mode == "pool":
        pool_width = width or 15
        pool_height = height or 15
        return f"{condition_id}\n({pool_width}x{pool_height})"
    if width and height:
        base = condition_id.replace("_25", "")
        return f"{base}\n({width}x{height})"
    return condition_id


def _load_manifest(experiment_dir: Path) -> dict[str, Any]:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return dict(normalize_export_keys(json.loads(manifest_path.read_text(encoding="utf-8"))))


def _load_summary_metrics(summary_path: Path) -> ReportData:
    data: ReportData = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for raw_row in csv.DictReader(handle):
            row = dict(normalize_export_keys(raw_row))
            condition = str(row["condition"])
            agent = str(row["agent_type"])
            if agent not in AGENTS:
                continue
            data.setdefault(condition, {})[agent] = {
                "train_succ": float(row["train_success_rate_mean"]),
                "eval_succ": float(row["eval_success_rate_mean"]),
                "tr_dec": float(row["train_mean_decisions_mean"]),
                "tr_steps": float(row["train_mean_steps_mean"]),
                "tau": float(row["train_mean_action_duration_mean"]),
                "train_r_m_observed": float(
                    row.get("train_R_m_observed_mean")
                    or (1.0 - float(row["train_success_rate_mean"]))
                ),
            }
    return data


def _ordered_conditions(data: ReportData) -> list[str]:
    available_conditions = list(data.keys())
    preferred = [condition for condition in DEFAULT_CONDS if condition in data]
    if not preferred:
        return available_conditions
    remaining = [condition for condition in available_conditions if condition not in preferred]
    return preferred + remaining


def _usable_conditions(data: ReportData, conditions: Sequence[str]) -> list[str]:
    usable: list[str] = []
    for condition in conditions:
        condition_agents = data.get(condition, {})
        if "DQN_LSTM_PRIMITIVE" not in condition_agents:
            continue
        if not any(agent in condition_agents for agent in MACRO_AGENTS):
            continue
        usable.append(condition)
    return usable


def _load_report_data(experiment_dir: Path) -> LoadedReportData:
    summary_path = experiment_dir / "summary_by_agent_condition.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing aggregated summary: {summary_path}")

    data = _load_summary_metrics(summary_path)
    manifest = _load_manifest(experiment_dir)
    usable_conditions = tuple(_usable_conditions(data, _ordered_conditions(data)))

    if not usable_conditions:
        raise ValueError(
            f"Cannot generate report figures from {summary_path}: "
            "need primitive rows plus at least one macro-agent row"
        )

    return LoadedReportData(
        data=data,
        manifest=manifest,
        conditions=usable_conditions,
    )


def _looks_like_experiment_dir(path: Path) -> bool:
    return (path / "summary_by_agent_condition.csv").exists()


def _candidate_results_root() -> Path:
    return Path(__file__).resolve().parents[3] / "research" / "results"


def _resolve_experiment_dir(arg_path: str | None) -> Path:
    if arg_path:
        return Path(arg_path).resolve()

    cwd = Path.cwd().resolve()
    if _looks_like_experiment_dir(cwd):
        return cwd

    results_root = _candidate_results_root()
    if results_root.exists():
        candidates = sorted(
            [path for path in results_root.iterdir() if path.is_dir() and _looks_like_experiment_dir(path)],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            try:
                _load_report_data(candidate)
                return candidate
            except (FileNotFoundError, ValueError):
                continue

    raise FileNotFoundError(
        "Could not find a research output directory. Run the script from a results folder or pass one explicitly, "
        "for example: python code/services/research/makeFigures.py research/results/lstm_macro_run"
    )


def _plot_context(loaded: LoadedReportData) -> PlotContext:
    return PlotContext(
        data=loaded.data,
        conditions=loaded.conditions,
        condition_labels={
            condition: _condition_label(condition, loaded.manifest)
            for condition in loaded.conditions
        },
    )


def _get(data: ReportData, condition: str, agent: str, key: str) -> float:
    return data[condition][agent][key]


def _present_agents(
    data: ReportData,
    conditions: Sequence[str],
    candidates: Sequence[str],
) -> list[str]:
    return [
        agent for agent in candidates
        if any(agent in data.get(condition, {}) for condition in conditions)
    ]


def _agents_present(context: PlotContext) -> list[str]:
    return _present_agents(context.data, context.conditions, AGENTS)


def _macro_agents_present(context: PlotContext) -> list[str]:
    return _present_agents(context.data, context.conditions, MACRO_AGENTS)


def _condition_marker(condition_id: str, index: int) -> str:
    return CONDITION_MARKERS.get(
        condition_id,
        FALLBACK_CONDITION_MARKERS[index % len(FALLBACK_CONDITION_MARKERS)],
    )


def _bar_offsets(count: int, width: float) -> list[float]:
    return [(index - (count - 1) / 2.0) * width for index in range(count)]


def _condition_legend_handles(context: PlotContext) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=_condition_marker(condition, index),
            color="w",
            markerfacecolor="#bbb",
            markeredgecolor="k",
            markersize=9,
            label=context.condition_labels[condition].replace("\n", " "),
        )
        for index, condition in enumerate(context.conditions)
    ]


def _agent_legend_handles(agents: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=AGENT_COLORS[agent],
            markeredgecolor="k",
            markersize=10,
            label=AGENT_LABELS[agent],
        )
        for agent in agents
    ]


def _plot_eval_success(context: PlotContext, out_path: Path) -> None:
    fig, ax = _new_figure((10.4, 6.4))
    x_positions = np.arange(len(context.conditions), dtype=float)
    agents = _agents_present(context)
    bar_width = min(0.22, 0.78 / max(len(agents), 1))

    for offset, agent in zip(_bar_offsets(len(agents), bar_width), agents):
        values = [
            100.0 * _get(context.data, condition, agent, "eval_succ")
            if agent in context.data.get(condition, {})
            else np.nan
            for condition in context.conditions
        ]
        ax.bar(
            x_positions + offset,
            values,
            bar_width,
            color=AGENT_COLORS[agent],
            label=AGENT_LABELS[agent],
            edgecolor="white",
            linewidth=0.6,
        )
        for x_coord, value in zip(x_positions + offset, values):
            if np.isnan(value):
                continue
            ax.text(x_coord, value + 2.0, f"{value:.0f}%", ha="center", va="bottom", fontsize=9.0)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([context.condition_labels[condition] for condition in context.conditions])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Evaluation success rate (% of evaluation episodes solved)")
    ax.set_title("Figure 1. Evaluation success rate by agent and condition", pad=14)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    _apply_common_axes_style(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(agents),
        frameon=False,
        fontsize=10.0,
        borderaxespad=0,
    )
    _save_figure(fig, out_path)


def _plot_decision_horizon_wall(context: PlotContext, out_path: Path) -> None:
    fig, ax = _new_figure((11.6, 7.8))
    for condition_index, condition in enumerate(context.conditions):
        marker = _condition_marker(condition, condition_index)
        for agent in AGENTS:
            if agent not in context.data.get(condition, {}):
                continue
            ax.scatter(
                _get(context.data, condition, agent, "tr_dec"),
                _get(context.data, condition, agent, "train_succ"),
                s=150,
                color=AGENT_COLORS[agent],
                marker=marker,
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )

    _configure_log_x_axis(ax)
    ax.set_xlabel("Mean decisions per training episode (log scale)")
    ax.set_ylabel("Training success rate (fraction of episodes solved)")
    ax.set_ylim(-0.07, 1.1)
    ax.set_xlim(6, 420)
    ax.set_title("Figure 2. The decision-horizon wall", pad=10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    _apply_common_axes_style(ax)

    condition_legend = ax.legend(
        handles=_condition_legend_handles(context),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8.8,
        title="Condition",
        ncol=2,
        **_legend_box_style(),
    )
    ax.add_artist(condition_legend)
    ax.legend(
        handles=_agent_legend_handles(_agents_present(context)),
        loc="upper right",
        bbox_to_anchor=(1.0, 0.915),
        fontsize=8.8,
        title="Agent",
        ncol=2,
        **_legend_box_style(),
    )
    _save_figure(fig, out_path)


def _plot_break_even(context: PlotContext, out_path: Path) -> None:
    fig, ax = _new_figure((10.2, 6.8))
    limits = (0.85, 11.0)
    points: list[tuple[float, float, bool, str, str]] = []
    max_threshold = 1.6

    for condition_index, condition in enumerate(context.conditions):
        marker = _condition_marker(condition, condition_index)
        primitive_decisions = _get(context.data, condition, "DQN_LSTM_PRIMITIVE", "tr_dec")
        primitive_steps = _get(context.data, condition, "DQN_LSTM_PRIMITIVE", "tr_steps")
        for agent in MACRO_AGENTS:
            if agent not in context.data.get(condition, {}):
                continue
            compression = primitive_decisions / _get(context.data, condition, agent, "tr_dec")
            risk_metric = _get(context.data, condition, agent, "train_r_m_observed")
            threshold = 1 + (
                (_get(context.data, condition, agent, "tr_steps") - primitive_steps) / primitive_steps
            ) + (DEFAULT_LAMBDA_FAILURE * risk_metric)
            solved = _get(context.data, condition, agent, "eval_succ") > 0.5
            points.append((compression, threshold, solved, marker, agent))
            max_threshold = max(max_threshold, threshold)

    y_max = max(1.6, float(np.ceil((max_threshold + 0.05) * 10.0) / 10.0))
    ax.fill_between(limits, limits, (0.2, 0.2), color="#2ca02c", alpha=0.07)
    ax.fill_between(limits, (y_max, y_max), limits, color="#d62728", alpha=0.07)
    ax.plot(limits, limits, "k--", lw=1, zorder=1)

    for compression, threshold, solved, marker, agent in points:
        ax.scatter(
            compression,
            threshold,
            s=210,
            marker=marker,
            facecolors=(AGENT_COLORS[agent] if solved else "white"),
            edgecolors=("black" if solved else AGENT_COLORS[agent]),
            linewidth=(1.0 if solved else 1.8),
            zorder=4,
        )

    ax.set_xlim(limits)
    ax.set_ylim(0.25, y_max)
    _configure_log_x_axis(ax)
    ax.set_xlabel(r"Decision compression ratio $C = H_\mathrm{p} / H_\mathrm{m}$ (log scale)")
    ax.set_ylabel(r"Break-even threshold $1 + O + R_m$ (dimensionless)")
    ax.set_title("Figure 4. Break-even prediction vs. observed outcome", pad=10)
    y_major = 0.2 if y_max > 1.8 else 0.1
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_major / 2.0))
    _apply_common_axes_style(ax)

    condition_legend = ax.legend(
        handles=_condition_legend_handles(context),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8.8,
        title="Condition",
        ncol=2,
        **_legend_box_style(),
    )
    ax.add_artist(condition_legend)

    agent_legend = ax.legend(
        handles=_agent_legend_handles(_macro_agents_present(context)),
        loc="upper right",
        bbox_to_anchor=(1.0, 0.90),
        fontsize=9.4,
        title="Agent",
        **_legend_box_style(),
    )
    ax.add_artist(agent_legend)

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#888",
                markeredgecolor="k",
                markersize=10,
                label="solved (filled)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor="#666666",
                markeredgewidth=1.8,
                markersize=10,
                label="failed (open)",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.768),
        fontsize=9.4,
        title="Outcome",
        **_legend_box_style(),
    )
    _save_figure(fig, out_path)


def _plot_macro_duration(context: PlotContext, out_path: Path) -> None:
    fig, ax = _new_figure((10.6, 6.6))
    x_positions = np.arange(len(context.conditions), dtype=float)
    agents = _macro_agents_present(context)
    bar_width = min(0.28, 0.78 / max(len(agents), 1))

    for offset, agent in zip(_bar_offsets(len(agents), bar_width), agents):
        values = [
            _get(context.data, condition, agent, "tau")
            if agent in context.data.get(condition, {})
            else np.nan
            for condition in context.conditions
        ]
        ax.bar(
            x_positions + offset,
            values,
            bar_width,
            color=AGENT_COLORS[agent],
            label=AGENT_LABELS[agent],
            edgecolor="white",
        )
        for x_coord, value in zip(x_positions + offset, values):
            if np.isnan(value):
                continue
            ax.text(x_coord, value + 0.08, f"{value:.2f}", ha="center", fontsize=9.2)

    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.text(
        len(context.conditions) - 0.65,
        1.05,
        r"$\tau = 1$" + "\n(no compression)",
        fontsize=9.0,
        va="bottom",
        bbox=_callout_box(),
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([context.condition_labels[condition] for condition in context.conditions])
    ax.set_ylabel(r"Mean macro length $\tau$ (primitive steps per decision)")
    ax.set_title(r"Figure 3. Realized macro length $\tau$ by condition", pad=12)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    _apply_common_axes_style(ax)
    ax.legend(frameon=False, fontsize=10.0, loc="upper left")
    _save_figure(fig, out_path)


def _resolve_output_dir(base_dir: Path, figures_dir: str | None) -> Path:
    output_dir = Path(figures_dir).resolve() if figures_dir else base_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _remove_legacy_outputs(output_dir: Path) -> None:
    for legacy_name in LEGACY_FIGURE_NAMES:
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()


def _figure_specs() -> tuple[tuple[str, FigureRenderer], ...]:
    return (
        ("fig1_success.png", _plot_eval_success),
        ("fig2_horizon_wall.png", _plot_decision_horizon_wall),
        ("fig3_tau.png", _plot_macro_duration),
        ("fig4_breakeven.png", _plot_break_even),
    )


def generate_report_figures(experiment_dir: str, figures_dir: str | None = None) -> list[str]:
    base_dir = Path(experiment_dir).resolve()
    context = _plot_context(_load_report_data(base_dir))
    output_dir = _resolve_output_dir(base_dir, figures_dir)
    _remove_legacy_outputs(output_dir)

    outputs: list[str] = []
    for filename, renderer in _figure_specs():
        output_path = output_dir / filename
        renderer(context, output_path)
        outputs.append(str(output_path))
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    experiment_dir = _resolve_experiment_dir(args[0] if args else None)
    outputs = generate_report_figures(str(experiment_dir))
    print("Wrote " + ", ".join(os.path.basename(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
