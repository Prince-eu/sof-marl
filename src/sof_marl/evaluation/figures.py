"""Report figures (EVALUATION.md section 4): learning curves, coordination-lift
bar chart, and a representative cash trajectory. The credit-model ROC/calibration
and SHAP figures were produced in Phase A (credit/train_credit.py, credit/explain.py).

The learning-curve y-axis is mean terminal firm value per training rollout, not
raw PPO reward: IPPO's, MAPPO's, and the single-agent baseline's rewards are on
different scales (MAPPO's includes the shared-reward term folded into each of the
four agents; the single-agent baseline's does not), so firm value -- an objective,
reward-scheme-independent measure -- is what makes the three curves comparable on
one axis. Figure captions state this and that the environment is a calibrated
simulation throughout (ARCHITECTURE.md, CLAUDE.md).

Usage:
    python -m sof_marl.evaluation.figures [--config config/train.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numpy.typing import NDArray

POLICY_LABELS = {
    "rule_based": "Rule-based",
    "single_agent_ppo": "Single-agent PPO",
    "ippo": "IPPO (independent)",
    "mappo": "MAPPO (coordinated)",
}
LEARNED_POLICY_COLORS = {
    "single_agent_ppo": "tab:blue",
    "ippo": "tab:orange",
    "mappo": "tab:green",
}


def _bucket_single_agent_curve(
    episode_timesteps: list[int],
    episode_firm_values: list[float],
    bucket_width: int,
    total_timesteps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_buckets = max(1, total_timesteps // bucket_width)
    buckets: list[list[float]] = [[] for _ in range(n_buckets)]
    for t, fv in zip(episode_timesteps, episode_firm_values, strict=True):
        b = min(int(t // bucket_width), n_buckets - 1)
        buckets[b].append(fv)
    x = (np.arange(1, n_buckets + 1) * bucket_width).astype(np.float64)
    y = np.array([np.mean(b) if b else np.nan for b in buckets], dtype=np.float64)
    return x, y


def _multiagent_curve(
    runs: list[dict[str, Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Returns (timesteps, mean_firm_value_across_seeds, std_across_seeds)."""
    n_points = min(len(run["learning_curve"]) for run in runs)
    x = np.array([runs[0]["learning_curve"][i]["timesteps"] for i in range(n_points)])
    y_per_seed = np.array(
        [[run["learning_curve"][i]["mean_firm_value"] for i in range(n_points)] for run in runs]
    )
    return x, y_per_seed.mean(axis=0), y_per_seed.std(axis=0)


def plot_learning_curves(results_dir: Path, ppo_total_timesteps: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ippo_runs = json.loads((results_dir / "learning_curve_ippo.json").read_text())
    x, y_mean, y_std = _multiagent_curve(ippo_runs)
    ax.plot(x, y_mean, color=LEARNED_POLICY_COLORS["ippo"], label=POLICY_LABELS["ippo"])
    ax.fill_between(
        x, y_mean - y_std, y_mean + y_std, color=LEARNED_POLICY_COLORS["ippo"], alpha=0.2
    )

    mappo_runs = json.loads((results_dir / "learning_curve_mappo.json").read_text())
    x, y_mean, y_std = _multiagent_curve(mappo_runs)
    ax.plot(x, y_mean, color=LEARNED_POLICY_COLORS["mappo"], label=POLICY_LABELS["mappo"])
    ax.fill_between(
        x, y_mean - y_std, y_mean + y_std, color=LEARNED_POLICY_COLORS["mappo"], alpha=0.2
    )

    sa_runs = json.loads((results_dir / "learning_curve_single_agent_ppo.json").read_text())
    bucket_width = int(ippo_runs[0]["learning_curve"][0]["timesteps"])  # = ppo.n_steps
    sa_curves = [
        _bucket_single_agent_curve(
            run["episode_timesteps"], run["episode_firm_values"], bucket_width, ppo_total_timesteps
        )
        for run in sa_runs
    ]
    x_sa = sa_curves[0][0]
    y_sa = np.array([c[1] for c in sa_curves])
    y_sa_mean = np.nanmean(y_sa, axis=0)
    y_sa_std = np.nanstd(y_sa, axis=0)
    ax.plot(
        x_sa,
        y_sa_mean,
        color=LEARNED_POLICY_COLORS["single_agent_ppo"],
        label=POLICY_LABELS["single_agent_ppo"],
    )
    ax.fill_between(
        x_sa,
        y_sa_mean - y_sa_std,
        y_sa_mean + y_sa_std,
        color=LEARNED_POLICY_COLORS["single_agent_ppo"],
        alpha=0.2,
    )

    eval_metrics = json.loads((results_dir / "eval_metrics.json").read_text())
    rule_based_fv = eval_metrics["policies"]["rule_based"]["aggregated"]["terminal_firm_value"][
        "mean"
    ]
    ax.axhline(
        rule_based_fv,
        color="black",
        linestyle="--",
        label=f"{POLICY_LABELS['rule_based']} (eval mean)",
    )

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean terminal firm value per rollout (USD, simulated)")
    ax.set_title("Learning curves: mean terminal firm value vs. training timesteps")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coordination_lift_bar_chart(results_dir: Path, out_path: Path) -> None:
    eval_metrics = json.loads((results_dir / "eval_metrics.json").read_text())
    policies = ["rule_based", "single_agent_ppo", "ippo", "mappo"]
    means = [
        eval_metrics["policies"][p]["aggregated"]["combined_objective"]["mean"] for p in policies
    ]
    stds = [
        eval_metrics["policies"][p]["aggregated"]["combined_objective"]["std"] for p in policies
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["gray", *[LEARNED_POLICY_COLORS[p] for p in policies[1:]]]
    ax.bar(
        [POLICY_LABELS[p] for p in policies],
        means,
        yerr=stds,
        color=colors,
        capsize=5,
    )
    ax.set_ylabel("Combined objective J (held-out evaluation episodes, simulated)")
    ax.set_title("Coordination lift: combined objective by policy")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_representative_cash_trajectory(results_dir: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    episode_index = 0

    rule_based = json.loads((results_dir / "eval_episodes_rule_based.json").read_text())
    series_by_policy: dict[str, list[float]] = {
        "rule_based": rule_based[episode_index]["cash_values"]
    }
    breaches_by_policy: dict[str, list[bool]] = {
        "rule_based": rule_based[episode_index]["solvency_breaches"]
    }
    for policy in ("single_agent_ppo", "ippo", "mappo"):
        data = json.loads((results_dir / f"eval_episodes_{policy}.json").read_text())
        first_seed = sorted(data.keys(), key=int)[0]
        series_by_policy[policy] = data[first_seed][episode_index]["cash_values"]
        breaches_by_policy[policy] = data[first_seed][episode_index]["solvency_breaches"]

    for policy, cash in series_by_policy.items():
        weeks = np.arange(len(cash))
        color = "black" if policy == "rule_based" else LEARNED_POLICY_COLORS[policy]
        ax.plot(weeks, cash, label=POLICY_LABELS[policy], color=color)
        breaches = breaches_by_policy[policy]
        breach_weeks = [w + 1 for w, b in enumerate(breaches) if b]
        if breach_weeks:
            ax.scatter(
                breach_weeks,
                [cash[w] for w in breach_weeks],
                color=color,
                marker="x",
                s=40,
                zorder=5,
            )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cash (USD, simulated)")
    ax.set_title(f"Representative 52-week cash trajectory (held-out episode {episode_index})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.yaml")
    args = parser.parse_args()
    train_cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    ppo_total_timesteps: int = train_cfg["ppo"]["total_timesteps"]

    results_dir = Path("reports/results")
    figures_dir = Path("reports/figures")

    plot_learning_curves(results_dir, ppo_total_timesteps, figures_dir / "learning_curves.png")
    plot_coordination_lift_bar_chart(results_dir, figures_dir / "coordination_lift.png")
    plot_representative_cash_trajectory(results_dir, figures_dir / "cash_trajectory.png")
    print(f"Wrote figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
