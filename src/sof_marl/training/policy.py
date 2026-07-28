"""Compact actor and critic networks for the custom multi-agent PPO trainer.

Two actor types, dispatched by each agent's native gymnasium action space
(``agents/spaces.py``): a diagonal-Gaussian policy for the three continuous (Box)
agents (liquidity, expenditure, capital_allocation), and an independent per-slot
Categorical policy for the discrete (MultiDiscrete) credit_risk agent.

Action bounds are enforced only at the environment-interaction boundary
(``clip_to_bounds``), never inside log-probability/entropy computations -- clamping
the *stored* action before computing its log-probability would break the PPO
importance-sampling ratio (``log_prob_new(a) / log_prob_old(a)`` must use the same
``a`` on both sides). Rollout collection stores the raw sampled action and clips
only the copy sent to ``env.step``.

Critics are a small MLP: ``training/multi_agent_ppo.py`` uses one independent
critic per agent (local observation input) for IPPO, and one centralized critic
with a per-agent output head (joint observation input) for MAPPO.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from numpy.typing import NDArray
from torch.distributions import Categorical, Normal

from sof_marl.agents import spaces as spaces_module
from sof_marl.env.calibration import EnvConfig

HIDDEN_SIZE = 64


def _mlp(in_dim: int, out_dim: int, hidden: int = HIDDEN_SIZE) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class GaussianActor(nn.Module):
    """Diagonal-Gaussian policy for a Box action space."""

    def __init__(self, obs_dim: int, low: NDArray[np.float32], high: NDArray[np.float32]) -> None:
        super().__init__()
        action_dim = len(low)
        self.mean_net = _mlp(obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.low: torch.Tensor
        self.high: torch.Tensor
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def log_prob_entropy(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def clip_to_bounds(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action, self.low, self.high)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.mean_net(obs), self.low, self.high)


class MultiDiscreteActor(nn.Module):
    """Independent per-slot Categorical policy for a MultiDiscrete action space."""

    def __init__(self, obs_dim: int, n_slots: int, n_categories: int) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.n_categories = n_categories
        self.logits_net = _mlp(obs_dim, n_slots * n_categories)

    def _logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.logits_net(obs).view(*obs.shape[:-1], self.n_slots, self.n_categories)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self._logits(obs))

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def log_prob_entropy(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def clip_to_bounds(self, action: torch.Tensor) -> torch.Tensor:
        return action  # discrete actions are always in-range by construction

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self._logits(obs), dim=-1)


class Critic(nn.Module):
    """Value network; ``n_heads`` outputs (1 per agent for a centralized critic)."""

    def __init__(self, obs_dim: int, n_heads: int = 1) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, n_heads)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def build_actor(agent: str, cfg: EnvConfig) -> nn.Module:
    """Dispatch to GaussianActor or MultiDiscreteActor per the agent's action space."""
    obs_space = spaces_module.observation_spaces(cfg)[agent]
    act_space = spaces_module.action_spaces(cfg)[agent]
    assert isinstance(obs_space, spaces.Box)
    obs_dim = int(obs_space.shape[0])

    if isinstance(act_space, spaces.Box):
        return GaussianActor(
            obs_dim, act_space.low.astype(np.float32), act_space.high.astype(np.float32)
        )
    if isinstance(act_space, spaces.MultiDiscrete):
        n_slots = len(act_space.nvec)
        n_categories = int(act_space.nvec[0])
        return MultiDiscreteActor(obs_dim, n_slots, n_categories)
    raise ValueError(f"Unsupported action space type for agent {agent!r}: {type(act_space)}")
