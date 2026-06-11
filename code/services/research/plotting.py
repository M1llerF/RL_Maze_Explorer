from __future__ import annotations

import os
from typing import Any

from services.research.schema import normalize_export_keys

AGENT_COLORS: dict[str, str] = {
    "DQN_LSTM_PRIMITIVE": "steelblue",
    "DQN_LSTM_NAIVE_MACROS": "tomato",
    "DQN_LSTM_ASTAR_MACROS": "darkorange",
    "DQN_LSTM_MOMENTUM_MACROS": "seagreen",
}


def generate_figures(out_dir: str, plot_data_dir: str) -> list[str]:
    """Generate PNG figures from plot-data CSVs. Returns list of created file paths."""
    try:
        import csv
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Research] matplotlib not available - skipping figure generation")
        return []

    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    created: list[str] = []

    def _read_csv(name: str) -> list[dict[str, Any]]:
        path = os.path.join(plot_data_dir, name)
        if not os.path.exists(path):
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return [dict(normalize_export_keys(row)) for row in csv.DictReader(f)]

    def _read_first_available(*names: str) -> list[dict[str, Any]]:
        for name in names:
            rows = _read_csv(name)
            if rows:
                return rows
        return []

    reward_rows = _read_csv("reward_curve.csv")
    success_rows = _read_csv("success_curve.csv")
    path_inefficiency_rows = _read_first_available(
        "path_inefficiency_curve.csv",
        "path_efficiency_curve.csv",
    )
    eval_bars = _read_csv("final_eval_bars.csv")

    conditions = list(dict.fromkeys(r["condition"] for r in reward_rows))

    def _plot_curve(
        rows: list[dict[str, Any]],
        cond: str,
        phase: str,
        value_col: str,
        ylabel: str,
        title: str,
        out_path: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        agents = list(
            dict.fromkeys(
                r["agent_type"] for r in rows if r["condition"] == cond and r["phase"] == phase
            )
        )
        for agent in agents:
            agent_rows = sorted(
                [
                    r
                    for r in rows
                    if r["condition"] == cond and r["agent_type"] == agent and r["phase"] == phase
                ],
                key=lambda r: int(r["episode"]),
            )
            if not agent_rows:
                continue
            runs = list(dict.fromkeys(r["run_id"] for r in agent_rows))
            for run in runs:
                run_rows = [r for r in agent_rows if r["run_id"] == run]
                xs = [int(r["episode"]) for r in run_rows]
                ys = [float(r[value_col]) for r in run_rows]
                color = AGENT_COLORS.get(agent, "grey")
                ax.plot(xs, ys, alpha=0.3, color=color, linewidth=0.8)
            # Mean rolling per episode across runs
            from collections import defaultdict

            ep_vals: dict[int, list[float]] = defaultdict(list)
            for r in agent_rows:
                ep_vals[int(r["episode"])].append(float(r[value_col]))
            xs_mean = sorted(ep_vals.keys())
            ys_mean = [sum(ep_vals[x]) / len(ep_vals[x]) for x in xs_mean]
            ax.plot(
                xs_mean,
                ys_mean,
                label=agent,
                color=AGENT_COLORS.get(agent, "grey"),
                linewidth=1.5,
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    for cond in conditions:
        for phase in ("training", "evaluation"):
            p = os.path.join(figures_dir, f"reward_curve_{cond}_{phase}.png")
            _plot_curve(reward_rows, cond, phase, "rolling_mean", "Rolling Reward", f"Reward - {cond} ({phase})", p)
            if os.path.exists(p):
                created.append(p)

    # Success rate from block CSV - one figure per condition per phase
    for cond in conditions:
        for phase in ("training", "evaluation"):
            phase_rows = [r for r in success_rows if r["condition"] == cond and r["phase"] == phase]
            if not phase_rows:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            agents = list(dict.fromkeys(r["agent_type"] for r in phase_rows))
            for agent in agents:
                agent_rows = sorted(
                    [r for r in phase_rows if r["agent_type"] == agent],
                    key=lambda r: int(r["episode_start"]),
                )
                xs = [int(r["episode_start"]) for r in agent_rows]
                ys = [float(r["success_rate"]) for r in agent_rows]
                ax.plot(
                    xs,
                    ys,
                    label=agent,
                    color=AGENT_COLORS.get(agent, "grey"),
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                )
            ax.set_xlabel("Episode (block start)")
            ax.set_ylabel("Block Success Rate")
            ax.set_title(f"Success Rate - {cond} ({phase})")
            ax.set_ylim(0, 1.05)
            ax.legend()
            fig.tight_layout()
            out_path = os.path.join(figures_dir, f"success_rate_{cond}_{phase}.png")
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            created.append(out_path)

    # Path inefficiency ratio
    for cond in conditions:
        p = os.path.join(figures_dir, f"path_inefficiency_{cond}.png")
        _plot_curve(
            path_inefficiency_rows,
            cond,
            "training",
            "rolling_mean",
            "Path Inefficiency Ratio (Steps / Optimal)",
            f"Path Inefficiency Ratio - {cond}",
            p,
        )
        if os.path.exists(p):
            created.append(p)

    # Final eval bar charts
    if eval_bars:
        for metric, label, fname in [
            ("mean_success_rate", "Eval Success Rate", "final_eval_success.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4))
            x_labels = [f"{r['agent_type']}\n{r['condition']}" for r in eval_bars]
            ys = [float(r[metric]) for r in eval_bars]
            colors = [AGENT_COLORS.get(r["agent_type"], "grey") for r in eval_bars]
            ax.bar(range(len(x_labels)), ys, color=colors)
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_ylabel(label)
            ax.set_title(label)
            fig.tight_layout()
            out_path = os.path.join(figures_dir, fname)
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            created.append(out_path)

    return created
