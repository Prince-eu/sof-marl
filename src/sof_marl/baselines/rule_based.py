"""Rule-based treasury policy (docs/ARCHITECTURE.md section 6.1).

A fixed-buffer, pay-on-due, draw-only-to-avoid-shortfall, fixed-risk-cutoff,
fixed-surplus-split heuristic: the kind of policy a competent treasurer without a
learned model would actually run. EVALUATION.md section 1: "a weak baseline is the
most common way a POC like this loses credibility" -- every parameter is a
documented choice in config/baselines.yaml, not a magic number, and none is tuned
against an evaluation result.

Unlike the learned policies (Phase D), this policy reads ``env.state`` and
``env.cfg`` directly rather than the normalized observation vectors -- a real
treasurer works from the actual books, not a feature encoding built for a neural
network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from sof_marl.env import calibration
from sof_marl.env.sme_treasury_env import SMETreasuryEnv


@dataclass(frozen=True)
class RuleBasedConfig:
    target_buffer_frac: float
    credit_risk_cutoff: float
    capital_allocation_split: tuple[float, float, float]
    spend_now_frac: float
    essential_bias: float


def load_rule_based_config(path: str | Path = "config/baselines.yaml") -> RuleBasedConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())["rule_based"]
    return RuleBasedConfig(
        target_buffer_frac=raw["target_buffer_frac"],
        credit_risk_cutoff=raw["credit_risk_cutoff"],
        capital_allocation_split=tuple(raw["capital_allocation_split"]),
        spend_now_frac=raw["spend_now_frac"],
        essential_bias=raw["essential_bias"],
    )


class RuleBasedPolicy:
    """Stateless heuristic policy over the four SMETreasuryEnv agents."""

    def __init__(self, config_path: str | Path = "config/baselines.yaml") -> None:
        self.config = load_rule_based_config(config_path)

    def act(self, env: SMETreasuryEnv) -> dict[str, NDArray[Any]]:
        return {
            "liquidity": self._liquidity_action(env),
            "credit_risk": self._credit_risk_action(env),
            "expenditure": self._expenditure_action(env),
            "capital_allocation": self._capital_allocation_action(env),
        }

    def _liquidity_action(self, env: SMETreasuryEnv) -> NDArray[np.float32]:
        """Maintain the calibrated buffer; pay payables on the due date; draw the
        line only to avoid a shortfall, repay it when there is slack; no
        receivables acceleration (the credit line is this policy's only lever for
        shortfall coverage)."""
        state, cfg = env.state, env.cfg
        buffer_target = calibration.buffer_target_dollars(
            state.week, cfg, self.config.target_buffer_frac
        )
        headroom = cfg.credit_line.limit - state.credit_drawn
        if state.cash < buffer_target and headroom > 0:
            credit_draw_frac = float(np.clip((buffer_target - state.cash) / headroom, 0.0, 1.0))
        elif state.credit_drawn > 0 and state.cash > buffer_target:
            credit_draw_frac = -float(
                np.clip((state.cash - buffer_target) / state.credit_drawn, 0.0, 1.0)
            )
        else:
            credit_draw_frac = 0.0
        return np.array(
            [self.config.target_buffer_frac, credit_draw_frac, 0.0, 0.0], dtype=np.float32
        )

    def _credit_risk_action(self, env: SMETreasuryEnv) -> NDArray[np.int64]:
        """Approve (plain, no premium) below the fixed cutoff; deny otherwise."""
        pending = env.pending_exposures()
        n = env.cfg.max_pending_credit_requests
        actions = np.zeros(n, dtype=np.int64)
        for i, exposure in enumerate(pending):
            actions[i] = 1 if exposure["predicted_prob"] < self.config.credit_risk_cutoff else 0
        return actions

    def _expenditure_action(self, env: SMETreasuryEnv) -> NDArray[np.float32]:
        return np.array([self.config.spend_now_frac, self.config.essential_bias], dtype=np.float32)

    def _capital_allocation_action(self, env: SMETreasuryEnv) -> NDArray[np.float32]:
        return np.array(self.config.capital_allocation_split, dtype=np.float32)
