"""Compact multi-agent PPO trainer shared by IPPO and MAPPO (ARCHITECTURE.md 6).

Both arms share this trainer; the only differences are (a) whether the critic is
independent per agent (IPPO, ``centralized_critic=False``) or centralized over the
joint observation with a per-agent output head (MAPPO,
``centralized_critic=True``), and (b) ``shared_reward_weight`` (0 for IPPO, > 0 for
MAPPO) -- already folded into each agent's reward inside
``agents/rewards.py::per_agent_rewards``, so this trainer just consumes whatever
reward the environment returns. Each of the four agents keeps its own actor
(matching its own observation/action space) and its own PPO clipped-surrogate
update; only the value function differs in structure between the two arms.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from numpy.typing import NDArray

from sof_marl.agents.spaces import AGENT_NAMES
from sof_marl.env.sme_treasury_env import SMETreasuryEnv
from sof_marl.training.policy import Critic, build_actor
from sof_marl.training.rollout import collect_rollout, compute_gae


@dataclass(frozen=True)
class PPOHyperparameters:
    total_timesteps: int
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    learning_rate: float
    clip_range: float
    ent_coef: float
    n_epochs: int
    vf_coef: float
    max_grad_norm: float


def load_ppo_hyperparameters(path: str | Path = "config/train.yaml") -> PPOHyperparameters:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())["ppo"]
    return PPOHyperparameters(
        total_timesteps=raw["total_timesteps"],
        n_steps=raw["n_steps"],
        batch_size=raw["batch_size"],
        gamma=raw["gamma"],
        gae_lambda=raw["gae_lambda"],
        learning_rate=raw["learning_rate"],
        clip_range=raw["clip_range"],
        ent_coef=raw["ent_coef"],
        n_epochs=raw["n_epochs"],
        vf_coef=raw["vf_coef"],
        max_grad_norm=raw["max_grad_norm"],
    )


class IndependentCritics:
    """One critic per agent, conditioned only on that agent's own observation (IPPO)."""

    def __init__(self, env: SMETreasuryEnv) -> None:
        self.critics = {agent: Critic(_obs_dim(env, agent)) for agent in AGENT_NAMES}

    def value_fn(self, obs: dict[str, NDArray[np.float32]]) -> dict[str, float]:
        out = {}
        with torch.no_grad():
            for agent in AGENT_NAMES:
                obs_t = torch.as_tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
                out[agent] = float(self.critics[agent](obs_t).item())
        return out

    def values(self, obs_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {agent: self.critics[agent](obs_batch[agent]).squeeze(-1) for agent in AGENT_NAMES}

    def parameters_for(self, agent: str) -> list[nn.Parameter]:
        return list(self.critics[agent].parameters())


class CentralizedCritic:
    """One critic over the joint (concatenated) observation, per-agent output head (MAPPO)."""

    def __init__(self, env: SMETreasuryEnv) -> None:
        joint_dim = sum(_obs_dim(env, agent) for agent in AGENT_NAMES)
        self.net = Critic(joint_dim, n_heads=len(AGENT_NAMES))

    def _joint(self, obs: dict[str, NDArray[np.float32]]) -> NDArray[np.float32]:
        return np.concatenate([obs[agent] for agent in AGENT_NAMES]).astype(np.float32)

    def value_fn(self, obs: dict[str, NDArray[np.float32]]) -> dict[str, float]:
        joint = torch.as_tensor(self._joint(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self.net(joint).squeeze(0)
        return {agent: float(out[i].item()) for i, agent in enumerate(AGENT_NAMES)}

    def values(self, obs_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        joint = torch.cat([obs_batch[agent] for agent in AGENT_NAMES], dim=-1)
        out = self.net(joint)
        return {agent: out[:, i] for i, agent in enumerate(AGENT_NAMES)}

    def parameters_for(self, agent: str) -> list[nn.Parameter]:
        return list(self.net.parameters())


CriticModel = IndependentCritics | CentralizedCritic


def _obs_dim(env: SMETreasuryEnv, agent: str) -> int:
    space = env.observation_space(agent)
    assert space.shape is not None
    return int(space.shape[0])


def _action_dtype(agent: str) -> torch.dtype:
    return torch.int64 if agent == "credit_risk" else torch.float32


def train_multi_agent_ppo(
    env_factory: Callable[[], SMETreasuryEnv],
    centralized_critic: bool,
    shared_reward_weight: float,
    seed: int,
    hp: PPOHyperparameters,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    """Train IPPO (centralized_critic=False) or MAPPO (True) for one seed.

    Returns trained actors/critic plus a learning-curve log (mean per-episode
    summed reward and terminal firm value per rollout) -- a training-progress
    diagnostic, not the officially reported metric. The headline combined
    objective J (docs/technical_report.md 4.2) is computed separately from
    deterministic held-out evaluation rollouts (evaluation/rollout.py, Phase E).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    budget = total_timesteps if total_timesteps is not None else hp.total_timesteps

    env = env_factory()
    env.shared_reward_weight = shared_reward_weight

    actors = {agent: build_actor(agent, env.cfg) for agent in AGENT_NAMES}
    critic_model: CriticModel = (
        CentralizedCritic(env) if centralized_critic else IndependentCritics(env)
    )

    actor_optimizers = {
        agent: torch.optim.Adam(actors[agent].parameters(), lr=hp.learning_rate)
        for agent in AGENT_NAMES
    }
    if centralized_critic:
        assert isinstance(critic_model, CentralizedCritic)
        critic_optimizers = {
            "__shared__": torch.optim.Adam(critic_model.net.parameters(), lr=hp.learning_rate)
        }
    else:
        critic_optimizers = {
            agent: torch.optim.Adam(critic_model.parameters_for(agent), lr=hp.learning_rate)
            for agent in AGENT_NAMES
        }

    obs, _ = env.reset(seed=seed)
    n_updates = max(1, budget // hp.n_steps)
    learning_curve: list[dict[str, float]] = []
    total_steps_done = 0
    start_time = time.time()

    for _update in range(n_updates):
        batch, obs = collect_rollout(env, actors, critic_model.value_fn, hp.n_steps, obs)
        total_steps_done += hp.n_steps

        advantages: dict[str, NDArray[np.float64]] = {}
        returns: dict[str, NDArray[np.float64]] = {}
        for agent in AGENT_NAMES:
            traj = batch.trajectories[agent]
            adv, ret = compute_gae(
                np.array(traj.rewards, dtype=np.float64),
                np.array(traj.values, dtype=np.float64),
                np.array(traj.dones, dtype=np.bool_),
                np.array(traj.bootstrap_values, dtype=np.float64),
                batch.final_bootstrap_value[agent],
                hp.gamma,
                hp.gae_lambda,
            )
            advantages[agent] = adv
            returns[agent] = ret

        indices = np.arange(hp.n_steps)
        for _epoch in range(hp.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, hp.n_steps, hp.batch_size):
                mb_idx = indices[start : start + hp.batch_size]
                obs_batch_mb = {
                    agent: torch.as_tensor(
                        np.array(batch.trajectories[agent].obs)[mb_idx], dtype=torch.float32
                    )
                    for agent in AGENT_NAMES
                }

                for agent in AGENT_NAMES:
                    traj = batch.trajectories[agent]
                    action_mb = torch.as_tensor(
                        np.array(traj.actions)[mb_idx], dtype=_action_dtype(agent)
                    )
                    old_log_prob_mb = torch.as_tensor(
                        np.array(traj.log_probs)[mb_idx], dtype=torch.float32
                    )
                    adv_mb = torch.as_tensor(advantages[agent][mb_idx], dtype=torch.float32)
                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                    new_log_prob, entropy = actors[agent].log_prob_entropy(
                        obs_batch_mb[agent], action_mb
                    )
                    ratio = torch.exp(new_log_prob - old_log_prob_mb)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - hp.clip_range, 1.0 + hp.clip_range) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -entropy.mean()
                    actor_loss = policy_loss + hp.ent_coef * entropy_loss

                    actor_optimizers[agent].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actors[agent].parameters(), hp.max_grad_norm)
                    actor_optimizers[agent].step()

                new_values = critic_model.values(obs_batch_mb)
                if centralized_critic:
                    assert isinstance(critic_model, CentralizedCritic)
                    value_loss = torch.stack(
                        [
                            F.mse_loss(
                                new_values[agent],
                                torch.as_tensor(returns[agent][mb_idx], dtype=torch.float32),
                            )
                            for agent in AGENT_NAMES
                        ]
                    ).mean()
                    critic_optimizers["__shared__"].zero_grad()
                    (hp.vf_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(critic_model.net.parameters(), hp.max_grad_norm)
                    critic_optimizers["__shared__"].step()
                else:
                    for agent in AGENT_NAMES:
                        v_loss = F.mse_loss(
                            new_values[agent],
                            torch.as_tensor(returns[agent][mb_idx], dtype=torch.float32),
                        )
                        critic_optimizers[agent].zero_grad()
                        (hp.vf_coef * v_loss).backward()
                        nn.utils.clip_grad_norm_(
                            critic_model.parameters_for(agent), hp.max_grad_norm
                        )
                        critic_optimizers[agent].step()

        if batch.episode_returns:
            learning_curve.append(
                {
                    "timesteps": float(total_steps_done),
                    "mean_episode_return": float(np.mean(batch.episode_returns)),
                    "mean_firm_value": float(np.mean(batch.episode_firm_values)),
                    "n_episodes": float(len(batch.episode_returns)),
                }
            )

    return {
        "actors": actors,
        "critic_model": critic_model,
        "learning_curve": learning_curve,
        "total_timesteps": total_steps_done,
        "wall_clock_seconds": time.time() - start_time,
        "seed": seed,
    }
