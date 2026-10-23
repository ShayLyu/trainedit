"""Bounded-residual PPO. Actions are tanh-squashed into [-1, 1]."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.config import (
    PPO_CLIP,
    PPO_ENT,
    PPO_EPOCHS,
    PPO_GAMMA,
    PPO_GRAD_NORM,
    PPO_HIDDEN,
    PPO_LAMBDA,
    PPO_LR,
    PPO_MINIBATCH,
    PPO_N_STEPS,
    PPO_VF,
)


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = PPO_HIDDEN) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)
        # Fixed noise. A learned std grows until stochastic rollouts drop the object
        # and the update walks a working mean off the reference.
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.5), requires_grad=False)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.body(obs)
        mu = self.mu(feat)
        std = self.log_std.clamp(-2.5, 0.5).exp().expand_as(mu)
        value = self.value(feat).squeeze(-1)
        return mu, std, value

    def log_prob(self, mu: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        u = _atanh(action)
        normal = Normal(mu, std)
        return normal.log_prob(u).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mu, std, value = self.forward(o)
            if deterministic:
                u = mu
            else:
                u = mu + std * torch.randn_like(mu)
            action = torch.tanh(u)
        return action.squeeze(0).numpy(), float(self.log_prob(mu, std, action).item()), float(value.item())


def _gae(rewards, values, dones, bootstrap: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in reversed(range(n)):
        next_value = bootstrap if t == n - 1 else values[t + 1]
        if dones[t]:
            next_value = 0.0
            next_nonterminal = 0.0
        else:
            next_nonterminal = 1.0
        delta = rewards[t] + PPO_GAMMA * next_value - values[t]
        last = delta + PPO_GAMMA * PPO_LAMBDA * next_nonterminal * last
        adv[t] = last
    returns = adv + values
    return adv, returns


def ppo_update(policy: ActorCritic, optimizer: torch.optim.Optimizer, batch: dict) -> None:
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
    old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    idx = np.arange(n)
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, n, PPO_MINIBATCH):
            mb = idx[start : start + PPO_MINIBATCH]
            if len(mb) < 2:
                continue
            mu, std, value = policy(obs[mb])
            logp = policy.log_prob(mu, std, actions[mb])
            ratio = torch.exp(logp - old_logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns[mb] - value).pow(2).mean()
            entropy = (0.5 * torch.log(2 * np.pi * np.e * std.pow(2))).sum(-1).mean()
            loss = policy_loss + PPO_VF * value_loss - PPO_ENT * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), PPO_GRAD_NORM)
            optimizer.step()


def collect_rollout(env, policy: ActorCritic, n_steps: int) -> tuple[dict, int]:
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
    obs = env.reset()
    for _ in range(n_steps):
        action, logp, value = policy.act(obs, deterministic=False)
        next_obs, reward, done, _info = env.step(action)
        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(reward)
        done_buf.append(done)
        val_buf.append(value)
        obs = env.reset() if done else next_obs
    bootstrap = 0.0 if done_buf[-1] else policy.act(obs, deterministic=True)[2]
    adv, returns = _gae(
        np.asarray(rew_buf, dtype=np.float64),
        np.asarray(val_buf, dtype=np.float64),
        np.asarray(done_buf, dtype=bool),
        bootstrap,
    )
    batch = {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(act_buf, dtype=np.float32),
        "logp": np.asarray(logp_buf, dtype=np.float32),
        "returns": returns.astype(np.float32),
        "adv": adv.astype(np.float32),
    }
    return batch, n_steps


class ResidualPolicy:
    """Callable policy used by evaluation rollouts."""

    def __init__(self, net: ActorCritic) -> None:
        self.net = net

    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _, _ = self.net.act(obs, deterministic=deterministic)
        return action


def make_policy(obs_dim: int, act_dim: int, seed: int) -> tuple[ActorCritic, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = ActorCritic(obs_dim, act_dim)
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR)
    return net, opt


def train_steps(env, net: ActorCritic, opt, budget: int, eval_fn) -> tuple[list[tuple[int, float]], int]:
    """Train until `budget` control steps, counting collection and eval steps.

    `eval_fn(policy) -> (success_rate, steps)`.
    """
    steps = 0
    curve: list[tuple[int, float]] = []
    policy = ResidualPolicy(net)
    sr, used = eval_fn(policy)
    steps += used
    curve.append((steps, sr))
    best_sr = sr
    best_state = copy.deepcopy(net.state_dict())
    while steps < budget:
        n = min(PPO_N_STEPS, budget - steps)
        if n < 32:
            break
        batch, used = collect_rollout(env, net, n)
        steps += used
        ppo_update(net, opt, batch)
        sr, used = eval_fn(policy)
        steps += used
        if sr + 1e-6 < best_sr:
            net.load_state_dict(best_state)
            sr = best_sr
        else:
            best_sr = sr
            best_state = copy.deepcopy(net.state_dict())
        curve.append((steps, sr))
    return curve, steps
