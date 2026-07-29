"""Single-agent Gymnasium wrapper around SMETreasuryEnv (ARCHITECTURE.md section 6.2).

One controller observes the concatenation of all four specialized agents'
observations and emits one combined action vector covering every decision the four
agents would otherwise split among themselves -- using the identical underlying
environment (dynamics, rewards, credit exposures) as the multi-agent baselines.
Tests whether decomposition into specialized agents helps at all.

Stable-Baselines3's PPO requires a single homogeneous action space (Box, Discrete,
or MultiDiscrete), but the four agents' native action spaces mix continuous (Box)
and discrete (MultiDiscrete) types. The combined action space here is therefore one
Box; the credit_risk agent's discrete per-slot decisions are represented as
continuous scores in [0, 1] and discretized in ``step`` (< 1/3 deny, < 2/3 approve,
else approve_with_premium) -- a standard technique for giving a single
continuous-action policy control over an otherwise-discrete sub-decision.

The combined reward is the unweighted sum of the four agents' base rewards
(``shared_reward_weight = 0``): a single controller that already sees and decides
everything has no separate identity for a coordination term to act against, unlike
IPPO/MAPPO's lambda*R_shared split (ARCHITECTURE.md section 4.5); summing the base
per-function rewards is the natural single-controller stand-in for optimizing the
firm holistically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import Env, spaces
from numpy.typing import NDArray

from sof_marl.agents.spaces import AGENT_NAMES
from sof_marl.env.sme_treasury_env import SMETreasuryEnv

CREDIT_RISK_SLOTS_ASSUMED = 4  # validated against cfg.max_pending_credit_requests in __init__
CREDIT_RISK_THRESHOLDS = (1.0 / 3.0, 2.0 / 3.0)  # deny | approve | approve_with_premium cut points


class SingleAgentTreasuryEnv(Env[NDArray[np.float32], NDArray[np.float32]]):
    """Gymnasium wrapper: one policy, the concatenated observation/action vectors."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path = "config/env.yaml",
        agents_config_path: str | Path = "config/agents.yaml",
        sector_config_paths: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._env = SMETreasuryEnv(
            config_path,
            agents_config_path,
            shared_reward_weight=0.0,
            sector_config_paths=sector_config_paths,
        )
        n_slots = self._env.cfg.max_pending_credit_requests
        if n_slots != CREDIT_RISK_SLOTS_ASSUMED:
            raise ValueError(
                "SingleAgentTreasuryEnv assumes max_pending_credit_requests == "
                f"{CREDIT_RISK_SLOTS_ASSUMED}; got {n_slots}. Update CREDIT_RISK_SLOTS_ASSUMED."
            )
        self._credit_risk_slots = n_slots

        obs_dims: list[int] = []
        for agent in AGENT_NAMES:
            agent_obs_space = self._env.observation_space(agent)
            assert isinstance(agent_obs_space, spaces.Box)
            obs_dims.append(int(agent_obs_space.shape[0]))
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(sum(obs_dims),), dtype=np.float32
        )

        liquidity_space = self._env.action_space("liquidity")
        expenditure_space = self._env.action_space("expenditure")
        capital_space = self._env.action_space("capital_allocation")
        assert isinstance(liquidity_space, spaces.Box)
        assert isinstance(expenditure_space, spaces.Box)
        assert isinstance(capital_space, spaces.Box)

        self._liquidity_dim = int(liquidity_space.shape[0])
        self._expenditure_dim = int(expenditure_space.shape[0])
        self._capital_dim = int(capital_space.shape[0])

        low = np.concatenate(
            [
                liquidity_space.low,
                np.zeros(self._credit_risk_slots, dtype=np.float32),
                expenditure_space.low,
                capital_space.low,
            ]
        )
        high = np.concatenate(
            [
                liquidity_space.high,
                np.ones(self._credit_risk_slots, dtype=np.float32),
                expenditure_space.high,
                capital_space.high,
            ]
        )
        self.action_space: spaces.Box = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        obs, infos = self._env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), infos[AGENT_NAMES[0]]

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        actions = self._split_action(np.asarray(action, dtype=np.float32))
        obs, rewards, terminations, truncations, infos = self._env.step(actions)
        combined_reward = float(sum(rewards.values()))
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        return self._flatten_obs(obs), combined_reward, terminated, truncated, infos[AGENT_NAMES[0]]

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close()

    def _flatten_obs(self, obs: dict[str, NDArray[np.float32]]) -> NDArray[np.float32]:
        return np.concatenate([obs[agent] for agent in AGENT_NAMES]).astype(np.float32)

    def _split_action(self, action: NDArray[np.float32]) -> dict[str, NDArray[Any]]:
        i = 0
        liquidity = action[i : i + self._liquidity_dim]
        i += self._liquidity_dim
        credit_scores = action[i : i + self._credit_risk_slots]
        i += self._credit_risk_slots
        expenditure = action[i : i + self._expenditure_dim]
        i += self._expenditure_dim
        capital = action[i : i + self._capital_dim]

        credit_decisions = np.digitize(credit_scores, bins=list(CREDIT_RISK_THRESHOLDS)).astype(
            np.int64
        )
        return {
            "liquidity": liquidity.astype(np.float32),
            "credit_risk": credit_decisions,
            "expenditure": expenditure.astype(np.float32),
            "capital_allocation": capital.astype(np.float32),
        }
