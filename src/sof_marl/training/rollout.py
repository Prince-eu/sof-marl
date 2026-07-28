"""Multi-agent rollout collection and GAE(lambda) advantage estimation.

``collect_rollout`` steps all four SMETreasuryEnv agents simultaneously for
``n_steps`` (spanning multiple 52-week episodes, auto-resetting), storing each
agent's raw trajectory. It is agnostic to IPPO vs MAPPO: the caller supplies a
``value_fn`` callable that maps an observation dict to a per-agent value dict,
which encapsulates whichever critic architecture (independent per-agent critics
for IPPO, or one centralized joint-observation critic for MAPPO) is in use --
see ``training/multi_agent_ppo.py``.

Every episode in this environment ends by time-limit truncation, never a true
absorbing termination (ARCHITECTURE.md section 1: a 52-week fiscal year, not a
firm that ceases to exist). GAE therefore always bootstraps through episode
boundaries using the critic's value of the actual terminal observation, per
Pardo et al. (2018), "Time Limits in Reinforcement Learning" -- never treating a
truncation as if the MDP had ended (which would systematically bias returns near
the horizon downward). ``compute_gae`` implements this explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from sof_marl.agents.spaces import AGENT_NAMES
from sof_marl.env.sme_treasury_env import SMETreasuryEnv

ValueFn = Callable[[dict[str, NDArray[np.float32]]], dict[str, float]]


@dataclass
class AgentTrajectory:
    obs: list[NDArray[np.float32]] = field(default_factory=list)
    actions: list[NDArray[np.float32]] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    bootstrap_values: list[float] = field(default_factory=list)  # meaningful only where done=True
    dones: list[bool] = field(default_factory=list)


@dataclass
class RolloutBatch:
    trajectories: dict[str, AgentTrajectory]
    final_bootstrap_value: dict[str, float]
    episode_returns: list[
        float
    ]  # completed episodes' combined (summed-over-agents) reward, for logging
    episode_firm_values: list[float]  # completed episodes' terminal firm value, for logging


def collect_rollout(
    env: SMETreasuryEnv,
    actors: dict[str, nn.Module],
    value_fn: ValueFn,
    n_steps: int,
    current_obs: dict[str, NDArray[np.float32]],
) -> tuple[RolloutBatch, dict[str, NDArray[np.float32]]]:
    """Collect n_steps of experience, auto-resetting on episode end.

    Returns the batch and the observation to pass into the next call (so
    collection is stateful/contiguous across successive PPO iterations).
    """
    trajectories = {agent: AgentTrajectory() for agent in AGENT_NAMES}
    episode_returns: list[float] = []
    episode_firm_values: list[float] = []
    episode_reward_acc = 0.0
    obs = current_obs

    for _ in range(n_steps):
        values = value_fn(obs)
        env_actions: dict[str, NDArray[np.float32] | NDArray[np.int64]] = {}
        raw_actions: dict[str, torch.Tensor] = {}
        log_probs: dict[str, float] = {}
        for agent in AGENT_NAMES:
            obs_t = torch.as_tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_t, log_prob_t = actors[agent].sample(obs_t)
            raw_actions[agent] = action_t.squeeze(0)
            log_probs[agent] = float(log_prob_t.item())
            clipped = actors[agent].clip_to_bounds(action_t).squeeze(0)
            env_actions[agent] = clipped.numpy()

        next_obs, rewards, terminations, truncations, infos = env.step(env_actions)
        done = any(terminations.values()) or any(truncations.values())
        episode_reward_acc += sum(rewards.values())

        bootstrap_values: dict[str, float] = {}
        if done:
            bootstrap_values = value_fn(next_obs)
            episode_returns.append(episode_reward_acc)
            episode_firm_values.append(infos[AGENT_NAMES[0]]["firm_value"])
            episode_reward_acc = 0.0
            next_obs, _ = env.reset()

        for agent in AGENT_NAMES:
            traj = trajectories[agent]
            traj.obs.append(obs[agent])
            action_np = raw_actions[agent].numpy()
            if isinstance(action_np, np.ndarray) and action_np.dtype != np.int64:
                action_np = action_np.astype(np.float32)
            traj.actions.append(action_np)
            traj.log_probs.append(log_probs[agent])
            traj.rewards.append(float(rewards[agent]))
            traj.values.append(float(values[agent]))
            traj.bootstrap_values.append(float(bootstrap_values.get(agent, 0.0)))
            traj.dones.append(done)

        obs = next_obs

    final_bootstrap_value = value_fn(obs)
    batch = RolloutBatch(trajectories, final_bootstrap_value, episode_returns, episode_firm_values)
    return batch, obs


def compute_gae(
    rewards: NDArray[np.float64],
    values: NDArray[np.float64],
    dones: NDArray[np.bool_],
    bootstrap_values: NDArray[np.float64],
    final_bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GAE(lambda) advantages and returns for one agent's rollout, bootstrapping
    through every episode boundary (see module docstring)."""
    n_steps = len(rewards)
    next_values = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        if dones[t]:
            next_values[t] = bootstrap_values[t]
        elif t == n_steps - 1:
            next_values[t] = final_bootstrap_value
        else:
            next_values[t] = values[t + 1]

    advantages = np.zeros(n_steps, dtype=np.float64)
    last_gae_lam = 0.0
    for t in reversed(range(n_steps)):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        continuation = 0.0 if dones[t] else last_gae_lam
        last_gae_lam = delta + gamma * gae_lambda * continuation
        advantages[t] = last_gae_lam
    returns = advantages + values
    return advantages, returns
