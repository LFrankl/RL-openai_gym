"""
CartPole-v1 — DQN (Deep Q-Network)
------------------------------------
相比 Q-Learning 表格法，DQN 用神经网络拟合 Q(s,a)，直接吃连续状态，
不需要手工离散化。核心技巧：

  1. Experience Replay   — 打破样本相关性，提升数据利用率
  2. Target Network      — 固定 TD 目标，稳定训练
  3. ε-greedy 衰减      — 前期探索，后期利用

网络结构: obs(4) → Linear(128) → ReLU → Linear(128) → ReLU → Q(2)
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# ── 超参数 ─────────────────────────────────────────────
EPISODES        = 600
BATCH_SIZE      = 64
GAMMA           = 0.99
ALPHA           = 1e-3          # Adam 学习率
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.995
REPLAY_SIZE     = 10_000        # 经验回放池容量
TARGET_UPDATE   = 10            # 每隔多少回合同步 target network
MIN_REPLAY      = 500           # 回放池至少积累多少条才开始训练

DEVICE = torch.device("cpu")
# DEVICE = torch.device("mps")  # Mac M4 启用 GPU 加速（取消注释即可）    # Mac x86 无 MPS，直接 CPU


# ── 神经网络 ────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 经验回放 ────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
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


# ── 训练 ────────────────────────────────────────────────
def train():
    env        = gym.make("CartPole-v1")
    obs_dim    = env.observation_space.shape[0]   # 4
    act_dim    = env.action_space.n               # 2

    policy_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
    replay    = ReplayBuffer(REPLAY_SIZE)
    epsilon   = EPSILON_START
    rewards   = []

    for ep in range(EPISODES):
        obs, _       = env.reset()
        total_reward = 0

        while True:
            # ε-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = int(policy_net(t).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(obs, action, reward, next_obs, float(done))
            obs           = next_obs
            total_reward += reward

            # 回放池够了才开始更新
            if len(replay) >= MIN_REPLAY:
                s, a, r, s_, d = replay.sample(BATCH_SIZE)

                # 当前 Q 值
                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                # TD 目标：用 target network 计算
                with torch.no_grad():
                    max_next_q = target_net(s_).max(dim=1).values
                    td_target  = r + GAMMA * max_next_q * (1.0 - d)

                loss = nn.functional.mse_loss(q_values, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards.append(total_reward)

        # 定期同步 target network
        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards[-50:])
            print(f"Episode {ep+1:4d} | avg(50): {avg:6.1f} | ε={epsilon:.3f} | buf={len(replay)}")

    env.close()
    return policy_net, rewards


# ── 评估 ────────────────────────────────────────────────
def evaluate(policy_net: QNetwork, episodes: int = 10):
    env   = gym.make("CartPole-v1")
    total = 0
    policy_net.eval()
    for _ in range(episodes):
        obs, _    = env.reset()
        ep_reward = 0
        while True:
            with torch.no_grad():
                t      = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = int(policy_net(t).argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total += ep_reward
    env.close()
    avg = total / episodes
    print(f"\n评估 {episodes} 回合 → 平均得分: {avg:.1f}  (满分 500)")
    return avg


if __name__ == "__main__":
    print("=== 训练 CartPole DQN ===")
    policy_net, rewards = train()

    print("\n=== 评估 ===")
    evaluate(policy_net)
