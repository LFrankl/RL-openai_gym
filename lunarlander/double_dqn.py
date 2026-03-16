"""
LunarLander-v3 — Double DQN
============================

【为什么需要 Double DQN？】

  标准 DQN 的 TD 目标：
    y = r + γ * max_a' Q_θ̄(s', a')

  问题：max_a' Q_θ̄(s', a') 用同一个网络（target_net）同时做两件事：
    ① 选择最优动作：a* = argmax_a' Q_θ̄(s', a')
    ② 评估该动作的价值：Q_θ̄(s', a*)

  当 Q 值估计有噪声时（训练初期尤为明显），max 操作会系统性地
  选出噪声偏高的动作，导致 Q 值被持续高估（Overestimation Bias）。
  高估的 Q 值作为训练目标，又进一步拉高 Q 值估计，形成正反馈循环。

  Hasselt et al. (2016) 的实验表明：在 Atari 游戏上，标准 DQN
  对某些游戏的 Q 值高估可达真实值的 10 倍以上。

【Double DQN 的修正】

  将"选择动作"和"评估价值"拆开到两个不同的网络：

    a* = argmax_a' Q_θ(s', a')      ← 用 policy_net（在线网络）选择动作
    y  = r + γ * Q_θ̄(s', a*)        ← 用 target_net（目标网络）评估价值

  直觉：即使在线网络对某个动作的 Q 值有噪声偏高，目标网络对该动作
  的评估值也不一定同样偏高（两个网络参数不同）。用目标网络来评估，
  相当于对高估做了一次"独立检验"，从而显著减少偏差。

  代码改动极小：只需修改 TD 目标的计算方式（约 2 行）。
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# ─────────────────────────────────────────────
# 超参数（与 DQN 保持一致，方便对比）
# ─────────────────────────────────────────────
EPISODES       = 600
BATCH_SIZE     = 64
GAMMA          = 0.99
LR             = 1e-3
EPSILON_START  = 1.0
EPSILON_END    = 0.01
EPSILON_DECAY  = 0.995
REPLAY_SIZE    = 50_000
TARGET_UPDATE  = 10
MIN_REPLAY     = 1_000

DEVICE = torch.device("cpu")
# DEVICE = torch.device("mps")  # Mac M4 启用 GPU 加速（取消注释即可）


class QNetwork(nn.Module):
    """与 dqn.py 完全相同的网络结构，方便对比。"""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32, device=DEVICE),
            torch.tensor(actions,            dtype=torch.long,    device=DEVICE),
            torch.tensor(rewards,            dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=DEVICE),
            torch.tensor(dones,              dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buf)


def train():
    env     = gym.make("LunarLander-v3")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay    = ReplayBuffer(REPLAY_SIZE)
    epsilon   = EPSILON_START
    rewards   = []

    for ep in range(EPISODES):
        obs, _       = env.reset()
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    t      = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = int(policy_net(t).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(obs, action, reward, next_obs, float(done))
            obs          = next_obs
            total_reward += reward

            if len(replay) >= MIN_REPLAY:
                s, a, r, s_, d = replay.sample(BATCH_SIZE)

                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            if len(replay) >= MIN_REPLAY:
                s, a, r, s_, d = replay.sample(BATCH_SIZE)

                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # Double DQN 的核心改动（对比标准 DQN）：
                    #
                    # 标准 DQN（注释掉的版本）：
                    #   max_next_q = target_net(s_).max(dim=1).values
                    #   （target_net 同时选择动作 + 评估价值）
                    #
                    # Double DQN：
                    #   步骤1：用 policy_net（在线网络）选择下一步最优动作
                    #   步骤2：用 target_net（目标网络）评估该动作的价值
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    # 步骤1：policy_net 决定哪个动作最好
                    next_actions = policy_net(s_).argmax(dim=1, keepdim=True)

                    # 步骤2：target_net 评估该动作的价值（而非自己选的动作）
                    next_q = target_net(s_).gather(1, next_actions).squeeze(1)

                    td_target = r + GAMMA * next_q * (1.0 - d)

                loss = nn.functional.mse_loss(q_values, td_target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards.append(total_reward)

        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % 50 == 0:
            avg    = np.mean(rewards[-50:])
            solved = "✓ 已解决" if avg >= 200 else ""
            print(f"Episode {ep+1:4d} | avg(50): {avg:7.1f} | ε={epsilon:.3f} {solved}")

    env.close()
    return policy_net, rewards


def evaluate(policy_net: QNetwork, episodes: int = 20, render: bool = True):
    env   = gym.make("LunarLander-v3", render_mode="human" if render else None)
    total = 0
    policy_net.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _    = env.reset()
            ep_reward = 0
            while True:
                t      = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = int(policy_net(t).argmax(dim=1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            total += ep_reward
    env.close()
    avg = total / episodes
    print(f"\n评估 {episodes} 回合 → 平均得分: {avg:.1f}  (>=200 视为解决)")
    return avg


if __name__ == "__main__":
    print("=== LunarLander Double DQN ===")
    print("【改进】选动作用 policy_net，评估价值用 target_net，消除高估偏差\n")
    policy_net, rewards = train()
    print("\n=== 评估 ===")
    evaluate(policy_net)
