"""The SME treasury Markov game as a PettingZoo ParallelEnv (ARCHITECTURE.md).

Four agents -- liquidity, credit_risk, expenditure, capital_allocation -- act
simultaneously each week over a shared 52-week episode. Weekly cash/AR/AP/financing
accounting is delegated to ``dynamics.step``; per-agent and shared rewards to
``agents.rewards``; calibration constants and the real-data credit exposure pool to
``calibration``. This module's own job is orchestration: interpreting each agent's
raw action, running the credit-exposure arrival/decision/resolution lifecycle
(ARCHITECTURE.md section 5), and building normalized observations
(ARCHITECTURE.md section 2).

The environment is a **calibrated simulation**: every constant it draws on is a
named, sourced value in config/env.yaml (ARCHITECTURE.md section 7), except the
credit-risk agent's exposure pool, which carries real SBA 7(a) loan outcomes and
the Phase-A model's real predicted default probabilities.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo import ParallelEnv

from sof_marl.agents import rewards as rewards_module
from sof_marl.agents import spaces as spaces_module
from sof_marl.env import calibration, dynamics

K_PROJECTION_WEEKS = 4  # lookahead for the shared "projected_cash_next_k" observation feature


@dataclass
class _ActiveExposure:
    amount: float
    true_label: int
    approved_with_premium: bool
    resolution_week: int


class SMETreasuryEnv(ParallelEnv[str, NDArray[np.float32], NDArray[np.int64]]):
    """PettingZoo ParallelEnv for the SME treasury Markov game."""

    metadata = {"render_modes": [], "name": "sme_treasury_v0", "is_parallelizable": True}

    def __init__(
        self,
        config_path: str | Path = "config/env.yaml",
        agents_config_path: str | Path = "config/agents.yaml",
        shared_reward_weight: float = 0.0,
        sector_config_paths: list[str] | None = None,
    ) -> None:
        super().__init__()
        # Domain randomization over sector profiles: if sector_config_paths is given,
        # a profile is sampled uniformly at each reset (config/sectors/*.yaml). Only
        # economic constants vary across sectors, so observation/action spaces are
        # identical -- the spaces below are built once and stay valid. When it is None
        # the env uses the single config_path (baseline behavior, unchanged).
        self._sector_cfgs: list[calibration.EnvConfig] = (
            [calibration.load_env_config(p) for p in sector_config_paths]
            if sector_config_paths
            else []
        )
        self._sector_names: list[str] = (
            [Path(p).stem for p in sector_config_paths] if sector_config_paths else []
        )
        self.cfg = (
            self._sector_cfgs[0] if self._sector_cfgs else calibration.load_env_config(config_path)
        )
        self.current_sector: str = self._sector_names[0] if self._sector_names else "general"
        self.agents_cfg = rewards_module.load_agents_config(agents_config_path)
        self.shared_reward_weight = shared_reward_weight

        self.possible_agents: list[str] = list(spaces_module.AGENT_NAMES)
        self.agents: list[str] = list(self.possible_agents)
        self._observation_spaces = spaces_module.observation_spaces(self.cfg)
        self._action_spaces = spaces_module.action_spaces(self.cfg)
        self._exposure_pool = calibration.build_or_load_credit_exposure_pool(self.cfg)

        self.rng: np.random.Generator = np.random.default_rng()
        self.state: dynamics.FirmState = dynamics.initial_state(self.cfg)
        self._episode_exposures: pd.DataFrame = pd.DataFrame()
        self._arrivals_by_week: dict[int, list[int]] = {}
        self._pending_queue: deque[int] = deque()
        self._active_exposures: dict[int, _ActiveExposure] = {}
        self._presented_this_week: list[int] = []

    # --- PettingZoo API ---

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self._action_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, dict[str, Any]]]:
        self.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        # Domain randomization: sample this episode's sector profile (if configured).
        if self._sector_cfgs:
            i = int(self.rng.integers(0, len(self._sector_cfgs)))
            self.cfg = self._sector_cfgs[i]
            self.current_sector = self._sector_names[i]
        self.state = dynamics.initial_state(self.cfg)

        self._episode_exposures = calibration.sample_episode_exposures(
            self._exposure_pool, self.cfg, self.rng
        )
        self._arrivals_by_week = {}
        for idx, row in self._episode_exposures.iterrows():
            self._arrivals_by_week.setdefault(int(row["arrival_week"]), []).append(int(idx))
        self._pending_queue = deque()
        self._active_exposures = {}
        self._presented_this_week = []
        self._advance_credit_queue()

        obs = self._build_all_observations()
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions: dict[str, NDArray[np.float32] | NDArray[np.int64]]) -> tuple[
        dict[str, NDArray[np.float32]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        old_state = self.state

        credit_decisions, credit_decision_ev = self._process_credit_decisions(
            np.asarray(actions["credit_risk"])
        )
        credit_income, credit_default_loss, credit_resolutions = self._resolve_due_exposures()
        outstanding_exposure_total = self._outstanding_exposure_total()

        liquidity_action = np.asarray(actions["liquidity"], dtype=np.float64)
        expenditure_action = np.asarray(actions["expenditure"], dtype=np.float64)
        capital_action = np.asarray(actions["capital_allocation"], dtype=np.float64)

        week_actions = dynamics.WeekActions(
            target_buffer_frac=float(liquidity_action[0]),
            credit_draw_frac=float(liquidity_action[1]),
            defer_payables_frac=float(liquidity_action[2]),
            accelerate_receivables_frac=float(liquidity_action[3]),
            spend_now_frac=float(expenditure_action[0]),
            capital_allocation=capital_action,
            credit_income=credit_income,
            credit_default_loss=credit_default_loss,
            essential_bias=float(expenditure_action[1]),
        )
        result = dynamics.step(old_state, week_actions, self.cfg, self.rng)
        self.state = result.state

        agent_rewards = rewards_module.per_agent_rewards(
            result,
            old_state,
            self.cfg,
            self.agents_cfg,
            outstanding_exposure_total,
            credit_decision_ev,
            self.shared_reward_weight,
        )
        self._advance_credit_queue()

        is_done = self.state.week >= self.cfg.horizon_weeks
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: is_done for agent in self.possible_agents}
        obs = self._build_all_observations()
        diagnostics: dict[str, Any] = {
            "solvency_breach": result.solvency_breach,
            "shortfall_severity": result.shortfall_severity,
            "firm_value": dynamics.firm_value(self.state, self.cfg),
            "shared_reward": agent_rewards["shared"],
            "financing_cost": result.financing_cost_total,
            "credit_decisions": credit_decisions,
            "credit_resolutions": credit_resolutions,
        }
        infos = {agent: dict(diagnostics) for agent in self.possible_agents}
        reward_out = {agent: agent_rewards[agent] for agent in self.possible_agents}

        if is_done:
            self.agents = []

        return obs, reward_out, terminations, truncations, infos

    def render(self) -> None:
        fv = dynamics.firm_value(self.state, self.cfg)
        print(f"week={self.state.week} cash={self.state.cash:.2f} firm_value={fv:.2f}")

    def close(self) -> None:
        return None

    # --- credit exposure lifecycle (ARCHITECTURE.md section 5) ---

    def _advance_credit_queue(self) -> None:
        week = self.state.week
        for row_id in self._arrivals_by_week.get(week, []):
            self._pending_queue.append(row_id)
        self._presented_this_week = []
        while (
            self._pending_queue
            and len(self._presented_this_week) < self.cfg.max_pending_credit_requests
        ):
            self._presented_this_week.append(self._pending_queue.popleft())

    def _process_credit_decisions(
        self, credit_action: NDArray[np.int64]
    ) -> tuple[list[dict[str, Any]], float]:
        """Record each presented exposure's decision and accumulate this week's
        decision-time expected value (the credit agent's reward-shaping signal).

        Returns (per-exposure records for evaluation metrics, total decision EV in
        dollars). The EV of an approved exposure uses the real model's predicted
        default probability observed at decision time:
        ``margin*(1-p) - w_lgd*LGD*p`` (with the premium margin when approved with
        premium); a denial contributes 0. See agents/rewards.py::credit_risk_reward.
        """
        rw = self.agents_cfg.credit_risk_rewards
        lgd = self.cfg.loss_given_default_frac
        decisions: list[dict[str, Any]] = []
        decision_ev = 0.0
        for slot, row_id in enumerate(self._presented_this_week):
            decision = int(credit_action[slot])
            row = self._episode_exposures.loc[row_id]
            predicted_prob = float(row["predicted_prob"])
            amount = float(row["exposure_amount"])
            decisions.append(
                {
                    "predicted_prob": predicted_prob,
                    "true_label": int(row["default"]),
                    "amount": amount,
                    "decision": decision,  # 0=deny, 1=approve, 2=approve_with_premium
                }
            )
            if decision == 0:  # deny -> no EV contribution
                continue
            bps = rw.margin_bps + (self.agents_cfg.credit_risk_premium_bps if decision == 2 else 0)
            decision_ev += (bps / 10000.0) * (1.0 - predicted_prob) * amount
            decision_ev -= rw.w_loss_given_default * lgd * predicted_prob * amount
            self._active_exposures[row_id] = _ActiveExposure(
                amount=amount,
                true_label=int(row["default"]),
                approved_with_premium=(decision == 2),
                resolution_week=self.state.week + self.cfg.exposure_resolution_weeks,
            )
        return decisions, decision_ev

    def _resolve_due_exposures(self) -> tuple[float, float, list[dict[str, Any]]]:
        """Resolve exposures due this week; return (income, loss, per-exposure records)."""
        week = self.state.week
        income = 0.0
        loss = 0.0
        resolutions: list[dict[str, Any]] = []
        resolved_ids = [
            rid for rid, exp in self._active_exposures.items() if exp.resolution_week == week
        ]
        for row_id in resolved_ids:
            exp = self._active_exposures.pop(row_id)
            if exp.true_label == 0:  # PIF: repaid, earn margin (+ premium if applicable)
                bps = self.agents_cfg.credit_risk_rewards.margin_bps
                if exp.approved_with_premium:
                    bps += self.agents_cfg.credit_risk_premium_bps
                exposure_income = exp.amount * bps / 10000.0
                exposure_loss = 0.0
            else:  # CHGOFF: default, lose loss_given_default_frac of notional
                exposure_income = 0.0
                exposure_loss = exp.amount * self.cfg.loss_given_default_frac
            income += exposure_income
            loss += exposure_loss
            resolutions.append(
                {
                    "true_label": exp.true_label,
                    "amount": exp.amount,
                    "approved_with_premium": exp.approved_with_premium,
                    "income": exposure_income,
                    "loss": exposure_loss,
                }
            )
        return income, loss, resolutions

    def _outstanding_exposure_total(self) -> float:
        return sum(exp.amount for exp in self._active_exposures.values())

    def pending_exposures(self) -> list[dict[str, float]]:
        """This week's presented credit exposures (predicted_prob, exposure_amount).

        Public, so baseline policies (e.g. baselines/rule_based.py) that legitimately
        need the real underlying exposure data -- not just the normalized
        observation encoding -- have a stable way to read it.
        """
        result: list[dict[str, float]] = []
        for row_id in self._presented_this_week:
            row = self._episode_exposures.loc[row_id]
            result.append(
                {
                    "predicted_prob": float(row["predicted_prob"]),
                    "exposure_amount": float(row["exposure_amount"]),
                }
            )
        return result

    # --- observations (ARCHITECTURE.md section 2) ---

    def _build_shared_observation(self) -> NDArray[np.float32]:
        scale = self.cfg.revenue.weekly_mean
        state = self.state
        headroom = self.cfg.credit_line.limit - state.credit_drawn
        net_weekly_flow = scale * (
            1.0 - self.cfg.expenses.fixed_weekly_frac - self.cfg.expenses.discretionary_weekly_frac
        )
        projected_cash = state.cash + K_PROJECTION_WEEKS * net_weekly_flow
        week_frac = state.week / self.cfg.horizon_weeks
        return np.array(
            [state.cash / scale, projected_cash / scale, headroom / scale, week_frac],
            dtype=np.float32,
        )

    def _build_liquidity_observation(self, shared: NDArray[np.float32]) -> NDArray[np.float32]:
        scale = self.cfg.revenue.weekly_mean
        state = self.state
        headroom = self.cfg.credit_line.limit - state.credit_drawn
        forecast = calibration.weekly_revenue_mu(state.week, self.cfg)
        agent_obs = np.concatenate(
            [
                [state.cash / scale],
                state.ar_buckets / scale,
                state.ap_due / scale,
                [headroom / scale, forecast / scale],
            ]
        ).astype(np.float32)
        return np.concatenate([agent_obs, shared])

    def _build_credit_risk_observation(self, shared: NDArray[np.float32]) -> NDArray[np.float32]:
        scale = self.cfg.revenue.weekly_mean
        pending = self.pending_exposures()
        slots: list[float] = []
        for i in range(self.cfg.max_pending_credit_requests):
            if i < len(pending):
                slots.extend(
                    [pending[i]["predicted_prob"], pending[i]["exposure_amount"] / scale, 1.0]
                )
            else:
                slots.extend([0.0, 0.0, 0.0])
        headroom = self.cfg.credit_line.limit - self.state.credit_drawn
        agent_obs = np.array(
            slots
            + [
                self.state.cash / scale,
                headroom / scale,
                self._outstanding_exposure_total() / scale,
            ],
            dtype=np.float32,
        )
        return np.concatenate([agent_obs, shared])

    def _build_expenditure_observation(self, shared: NDArray[np.float32]) -> NDArray[np.float32]:
        scale = self.cfg.revenue.weekly_mean
        state = self.state
        mu = calibration.weekly_revenue_mu(state.week, self.cfg)
        discretionary_available = (
            self.cfg.expenses.discretionary_weekly_frac * mu + state.discretionary_carryover
        )
        season = 2.0 * np.pi * (state.week % 52) / 52.0
        agent_obs = np.array(
            [
                discretionary_available / scale,
                state.cash / scale,
                float(state.ap_due.sum()) / scale,
                float(np.sin(season)),
                float(np.cos(season)),
                state.ops_health,
            ],
            dtype=np.float32,
        )
        return np.concatenate([agent_obs, shared])

    def _build_capital_allocation_observation(
        self, shared: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        scale = self.cfg.revenue.weekly_mean
        state = self.state
        buffer_target = calibration.buffer_target_dollars(
            state.week, self.cfg, state.buffer_frac_multiplier
        )
        surplus = max(0.0, state.cash - buffer_target)
        agent_obs = np.array(
            [
                surplus / scale,
                state.term_debt / scale,
                self.cfg.term_debt_apr,
                state.reserve / scale,
                state.invested / scale,
                self.cfg.investment_return.weekly_mean,
            ],
            dtype=np.float32,
        )
        return np.concatenate([agent_obs, shared])

    def _build_all_observations(self) -> dict[str, NDArray[np.float32]]:
        shared = self._build_shared_observation()
        return {
            "liquidity": self._build_liquidity_observation(shared),
            "credit_risk": self._build_credit_risk_observation(shared),
            "expenditure": self._build_expenditure_observation(shared),
            "capital_allocation": self._build_capital_allocation_observation(shared),
        }
