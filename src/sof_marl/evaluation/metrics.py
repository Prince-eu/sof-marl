"""The full evaluation metric suite and the coordination-lift test (EVALUATION.md).

Every metric listed in EVALUATION.md section 2 is computed here from
``EpisodeRecord`` data (``evaluation/rollout.py``); none is selected or omitted
for being unflattering. The combined objective J and its weights are
pre-registered in ``config/train.yaml`` (not tuned to results). The coordination
lift is a paired Wilcoxon signed-rank test across the shared evaluation episodes:
for each episode, a policy's value is the mean of that metric over its training
seeds (rule-based has no seed axis and is treated as a single "seed"), and the
pairing dimension is the evaluation episode -- the same environment shocks
(revenue draws, credit exposures) presented to every policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy import stats

from sof_marl.env.calibration import EnvConfig, load_env_config, weekly_revenue_mu
from sof_marl.evaluation.rollout import EpisodeRecord

EpisodesBySeed = dict[int, list[EpisodeRecord]]


def episode_return(record: EpisodeRecord) -> float:
    v0 = record.firm_values[0]
    return (record.firm_values[-1] - v0) / v0 if v0 else 0.0


def episode_volatility(record: EpisodeRecord) -> float:
    v0 = record.firm_values[0]
    diffs = np.diff(np.array(record.firm_values))
    return float(np.std(diffs)) / v0 if v0 else 0.0


def episode_solvency_breach_rate(record: EpisodeRecord) -> float:
    return float(np.mean(record.solvency_breaches)) if record.solvency_breaches else 0.0


def episode_max_drawdown(record: EpisodeRecord) -> float:
    return float(np.min(record.cash_values))


def episode_financing_cost(record: EpisodeRecord) -> float:
    return float(np.sum(record.financing_costs))


def episode_buffer_days(record: EpisodeRecord, cfg: EnvConfig) -> NDArray[np.float64]:
    """Per-week buffer-day coverage: cash / calibrated daily operating outflow."""
    days = []
    for week_idx, cash in enumerate(record.cash_values[1:]):
        mu = weekly_revenue_mu(week_idx, cfg)
        weekly_outflow = (
            cfg.expenses.fixed_weekly_frac + cfg.expenses.discretionary_weekly_frac
        ) * mu
        daily_outflow = weekly_outflow / 7.0
        days.append(cash / daily_outflow if daily_outflow > 0 else 0.0)
    return np.array(days, dtype=np.float64)


def combined_objective(record: EpisodeRecord, kappa: float, lam: float) -> float:
    """J = r - kappa*s - lambda*B (docs/technical_report.md section 4.2)."""
    r = episode_return(record)
    s = episode_volatility(record)
    breach_rate = episode_solvency_breach_rate(record)
    return r - kappa * s - lam * breach_rate


def credit_metrics(records: list[EpisodeRecord]) -> dict[str, float]:
    """Approval precision/recall (approve ~ predicted non-default), realized
    default/loss rate, and portfolio yield net of losses, pooled over episodes."""
    all_decisions = [d for r in records for d in r.credit_decisions]
    all_resolutions = [res for r in records for res in r.credit_resolutions]

    true_positive = sum(1 for d in all_decisions if d["decision"] != 0 and d["true_label"] == 0)
    false_positive = sum(1 for d in all_decisions if d["decision"] != 0 and d["true_label"] == 1)
    false_negative = sum(1 for d in all_decisions if d["decision"] == 0 and d["true_label"] == 0)
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else float("nan")
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else float("nan")
    )

    total_amount = sum(res["amount"] for res in all_resolutions)
    total_loss = sum(res["loss"] for res in all_resolutions)
    total_income = sum(res["income"] for res in all_resolutions)
    n_defaults = sum(1 for res in all_resolutions if res["true_label"] == 1)

    return {
        "approval_precision": precision,
        "approval_recall": recall,
        "realized_default_rate": (
            n_defaults / len(all_resolutions) if all_resolutions else float("nan")
        ),
        "realized_loss_rate_dollar": total_loss / total_amount if total_amount else float("nan"),
        "portfolio_yield_net": (
            (total_income - total_loss) / total_amount if total_amount else float("nan")
        ),
        "n_decisions": float(len(all_decisions)),
        "n_resolutions": float(len(all_resolutions)),
    }


def _as_episodes_by_seed(episodes: EpisodesBySeed | list[EpisodeRecord]) -> EpisodesBySeed:
    return episodes if isinstance(episodes, dict) else {0: episodes}


def summarize_policy(
    episodes: EpisodesBySeed | list[EpisodeRecord], cfg: EnvConfig, kappa: float, lam: float
) -> dict[str, Any]:
    """Mean +/- std across training seeds (rule-based: across its one pseudo-seed,
    i.e. no seed-variance axis, since it is a fixed heuristic, not learned)."""
    episodes_by_seed = _as_episodes_by_seed(episodes)

    per_seed_metrics = []
    for seed, records in sorted(episodes_by_seed.items()):
        returns = [episode_return(r) for r in records]
        vols = [episode_volatility(r) for r in records]
        breaches = [episode_solvency_breach_rate(r) for r in records]
        objectives = [combined_objective(r, kappa, lam) for r in records]
        buffer_days_all = np.concatenate([episode_buffer_days(r, cfg) for r in records])
        # The representative firm starts with $0 invested (config/env.yaml
        # initial.invested), so a growth *ratio* against that baseline is
        # ill-defined (divide-by-near-zero). Report the dollar change in the
        # invested balance normalized by weekly revenue instead, consistent
        # with how every other dollar-valued reward/metric term in this
        # project is scaled (agents/rewards.py).
        invested_growth_norm = [
            (r.invested_values[-1] - r.invested_values[0]) / cfg.revenue.weekly_mean
            for r in records
        ]
        cred = credit_metrics(records)

        per_seed_metrics.append(
            {
                "seed": seed,
                "return_r": float(np.mean(returns)),
                "volatility_s": float(np.mean(vols)),
                "solvency_breach_rate": float(np.mean(breaches)),
                "max_drawdown": float(np.mean([episode_max_drawdown(r) for r in records])),
                "financing_cost": float(np.mean([episode_financing_cost(r) for r in records])),
                "terminal_firm_value": float(np.mean([r.firm_values[-1] for r in records])),
                "risk_adjusted_return": float(
                    np.mean([ret - kappa * v for ret, v in zip(returns, vols, strict=True)])
                ),
                "combined_objective": float(np.mean(objectives)),
                "buffer_days_mean": float(np.mean(buffer_days_all)),
                "buffer_days_p10": float(np.percentile(buffer_days_all, 10)),
                "invested_capital_growth_norm": float(np.mean(invested_growth_norm)),
                **cred,
            }
        )

    keys = [k for k in per_seed_metrics[0] if k != "seed"]
    aggregated = {
        k: {
            "mean": float(np.nanmean([m[k] for m in per_seed_metrics])),
            "std": float(np.nanstd([m[k] for m in per_seed_metrics])),
        }
        for k in keys
    }
    return {
        "per_seed": per_seed_metrics,
        "aggregated": aggregated,
        "n_seeds": len(per_seed_metrics),
    }


def per_episode_seed_averaged(
    episodes: EpisodesBySeed | list[EpisodeRecord], metric_fn: Any
) -> NDArray[np.float64]:
    """metric_fn(EpisodeRecord) -> float; returns length-n_episodes array, each
    entry the mean over training seeds of metric_fn for that episode index."""
    episodes_by_seed = _as_episodes_by_seed(episodes)
    seeds = sorted(episodes_by_seed.keys())
    n_episodes = len(episodes_by_seed[seeds[0]])
    out = np.zeros(n_episodes)
    for ep_idx in range(n_episodes):
        out[ep_idx] = np.mean([metric_fn(episodes_by_seed[s][ep_idx]) for s in seeds])
    return out


def coordination_lift(
    mappo_episodes: EpisodesBySeed,
    other_episodes: EpisodesBySeed | list[EpisodeRecord],
    kappa: float,
    lam: float,
    alpha: float,
) -> dict[str, Any]:
    def obj_fn(r: EpisodeRecord) -> float:
        return combined_objective(r, kappa, lam)

    mappo_obj = per_episode_seed_averaged(mappo_episodes, obj_fn)
    other_obj = per_episode_seed_averaged(other_episodes, obj_fn)
    mappo_breach = per_episode_seed_averaged(mappo_episodes, episode_solvency_breach_rate)
    other_breach = per_episode_seed_averaged(other_episodes, episode_solvency_breach_rate)

    diff_obj = mappo_obj - other_obj
    if np.allclose(diff_obj, 0.0):
        wilcoxon_stat, p_value = float("nan"), 1.0
    else:
        wilcoxon_stat, p_value = stats.wilcoxon(diff_obj)
    effect_size = (
        float(np.mean(diff_obj) / np.std(diff_obj)) if np.std(diff_obj) > 0 else float("nan")
    )

    return {
        "delta_combined_objective_mean": float(np.mean(diff_obj)),
        "delta_solvency_breach_rate_mean": float(np.mean(mappo_breach - other_breach)),
        "wilcoxon_statistic": float(wilcoxon_stat),
        "p_value": float(p_value),
        "effect_size_matched_pairs_d": effect_size,
        "significant_at_alpha": bool(p_value < alpha) if not np.isnan(p_value) else False,
    }


def build_evaluation_summary(
    rule_based_episodes: list[EpisodeRecord],
    single_agent_episodes: EpisodesBySeed,
    ippo_episodes: EpisodesBySeed,
    mappo_episodes: EpisodesBySeed,
    cfg_path: str = "config/train.yaml",
    env_cfg_path: str = "config/env.yaml",
) -> dict[str, Any]:
    train_cfg: dict[str, Any] = yaml.safe_load(Path(cfg_path).read_text())
    kappa: float = train_cfg["objective"]["kappa_volatility"]
    lam: float = train_cfg["objective"]["lambda_solvency"]
    alpha: float = train_cfg["significance"]["alpha"]
    cfg = load_env_config(env_cfg_path)

    policies = {
        "rule_based": summarize_policy(rule_based_episodes, cfg, kappa, lam),
        "single_agent_ppo": summarize_policy(single_agent_episodes, cfg, kappa, lam),
        "ippo": summarize_policy(ippo_episodes, cfg, kappa, lam),
        "mappo": summarize_policy(mappo_episodes, cfg, kappa, lam),
    }
    coordination = {
        "mappo_vs_rule_based": coordination_lift(
            mappo_episodes, rule_based_episodes, kappa, lam, alpha
        ),
        "mappo_vs_single_agent": coordination_lift(
            mappo_episodes, single_agent_episodes, kappa, lam, alpha
        ),
        "mappo_vs_ippo": coordination_lift(mappo_episodes, ippo_episodes, kappa, lam, alpha),
    }

    return {
        "objective_weights": {"kappa_volatility": kappa, "lambda_solvency": lam},
        "n_episodes": len(rule_based_episodes),
        "policies": policies,
        "coordination_lift": coordination,
    }
