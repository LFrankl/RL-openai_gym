"""
LunarLander-v3 — Dueling DQN
==============================

【为什么需要 Dueling DQN？】

  标准 DQN 的网络直接输出 Q(s, a)，即"在状态 s 下执行动作 a 的价值"。
  但很多时候，某些状态下无论执行什么动作，结果都差不多（比如飞船已经
  安全悬停时，当前帧做什么动作影响不大）。

  这时网络需要分别学好"这个状态本身有多好"和"在这个状态下哪个动作
  相对更好"，但标准网络把两件事混在一起学，效率低。

【Dueling DQN 的网络结构分解】

  Wang et al. (2016) 提出将 Q 值分解为两部分：

    Q(s, a) = V(s) + A(s, a)

  其中：
    V(s)    = 状态价值函数（与动作无关，标量）
              "不管做什么，这个状态本身值多少分"
    A(s, a) = 优势函数（Advantage Function）
              "相比平均水平，执行动作 a 有多好/差"
              A(s, a) = Q(s, a) - V(s)

  网络结构：
    共享主干（特征提取）
         ↓
    ┌────┴────┐
    V(s)     A(s, a)    ← 两个独立的输出头
    (1维)    (4维)
         ↓
    合并：Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

  减去 mean_a[A(s,a)] 是为了让分解唯一（否则 V 和 A 可以任意偏移抵消）。

【为什么有效？】

  在很多状态下，A(s, a) 的各动作差异很小（某些动作几乎等价），
  网络只需要精确学好 V(s)，A 头输出接近 0 即可。这让：

  1. V(s) 得到更充分的学习（因为每次更新都会影响所有动作的 Q 值）
  2. A(s, a) 更专注于学习"差异"而非绝对值
  3. 在动作较多、状态复杂的任务上收敛更快、更稳定

  Dueling 与 Double DQN 是正交的改进，可以同时使用
  （本文件同时使用了 Double DQN 的 TD 目标计算方式）。
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# ─────────────────────────────────────────────
# 超参数
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


# ─────────────────────────────────────────────
# Dueling 网络结构
# ─────────────────────────────────────────────
class DuelingQNetwork(nn.Module):
    """
    网络拓扑：

    输入(8) → 共享主干 → 隐藏特征(256)
                              ↓
                    ┌─────────┴─────────┐
              V 头：Linear(256→1)   A 头：Linear(256→4)
                    └─────────┬─────────┘
                              ↓
              Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        # 共享的特征提取层：两个头共用这部分的梯度
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # 价值头：输出标量 V(s)，代表当前状态的基础价值
        self.value_head = nn.Linear(256, 1)

        # 优势头：输出向量 A(s, ·)，代表每个动作相对于平均水平的优势
        self.advantage_head = nn.Linear(256, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 共享特征提取
        feat = self.backbone(x)

        # 2. 分别通过两个头
        V = self.value_head(feat)           # shape: (batch, 1)
        A = self.advantage_head(feat)       # shape: (batch, act_dim)

        # 3. 合并：减去 A 的均值，保证分解唯一性
        #    Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]
        #    这样 mean_a Q(s,a) = V(s)，语义一致
        Q = V + (A - A.mean(dim=1, keepdim=True))

        return Q  # shape: (batch, act_dim)


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

    # 使用 Dueling 网络，其余训练流程与 Double DQN 完全相同
    policy_net = DuelingQNetwork(obs_dim, act_dim).to(DEVICE)
    target_net = DuelingQNetwork(obs_dim, act_dim).to(DEVICE)
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

                # 当前 Q 值（Dueling 网络自动分解 V+A 并合并）
                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Double DQN 的 TD 目标（Dueling + Double 组合）：
                    # policy_net 选动作，target_net 评估价值
                    next_actions = policy_net(s_).argmax(dim=1, keepdim=True)
                    next_q       = target_net(s_).gather(1, next_actions).squeeze(1)
                    td_target    = r + GAMMA * next_q * (1.0 - d)

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


def evaluate(policy_net: DuelingQNetwork, episodes: int = 20):
    env   = gym.make("LunarLander-v3")
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
    print("=== LunarLander Dueling DQN ===")
    print("【改进】网络分解为 V(s) + A(s,a)，同时沿用 Double DQN 的 TD 目标\n")
    policy_net, rewards = train()
    print("\n=== 评估 ===")
    evaluate(policy_net)
