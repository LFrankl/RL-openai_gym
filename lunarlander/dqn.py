"""
LunarLander-v3 — 基础 DQN（Deep Q-Network）
============================================

【环境说明】
  飞船从屏幕上方随机位置下落，目标是稳稳降落在两个黄旗之间的着陆垫上。

  观测（8维连续）：
    [0] x 坐标          [1] y 坐标
    [2] x 速度          [3] y 速度
    [4] 角度            [5] 角速度
    [6] 左脚是否触地    [7] 右脚是否触地

  动作（4个离散）：
    0 = 什么都不做
    1 = 点火左侧引擎（机体向右转）
    2 = 点火主引擎（向上推力）
    3 = 点火右侧引擎（机体向左转）

  奖励设计（稠密奖励）：
    - 靠近着陆垫：+奖励（最多约 +140）
    - 远离着陆垫：-奖励
    - 每次点火引擎：-0.3（节约燃料）
    - 坠毁：-100
    - 成功着陆：+100
    - 双腿触地：每腿 +10
  总分 >= 200 视为"解决"

【DQN 核心思路】
  用神经网络 Q_θ(s, a) 逼近最优动作价值函数 Q*(s, a)。
  训练目标：最小化 TD 误差的均方值：

    L(θ) = E[(r + γ max_a' Q_θ̄(s', a') - Q_θ(s, a))²]

  其中 θ̄ 是"目标网络"的参数，每隔 C 步从 θ 同步一次。

【已知问题：Q 值高估（Overestimation Bias）】
  标准 DQN 使用同一个网络来"选择动作"和"评估动作价值"：

    y = r + γ * max_a' Q_θ̄(s', a')

  max 操作会系统性地选择 Q 值偏高的动作（因为估计误差是随机的，
  max 总倾向于选中噪声偏高的那个）。这会导致 Q 值被持续高估，
  训练后期可能出现不稳定或过早收敛到次优策略。

  → 这个问题由 Double DQN 解决（见 double_dqn.py）
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
EPISODES       = 600       # 训练总回合数
BATCH_SIZE     = 64        # 每次从回放池采样的数量
GAMMA          = 0.99      # 折扣因子：越大越重视未来奖励
LR             = 1e-3      # Adam 学习率
EPSILON_START  = 1.0       # 初始探索率（100% 随机）
EPSILON_END    = 0.01      # 最低探索率
EPSILON_DECAY  = 0.995     # 每回合乘以该系数衰减
REPLAY_SIZE    = 50_000    # 经验回放池容量
TARGET_UPDATE  = 10        # 每隔多少回合同步目标网络
MIN_REPLAY     = 1_000     # 回放池至少积累多少条才开始训练

DEVICE = torch.device("cpu")


# ─────────────────────────────────────────────
# Q 网络
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    输入：8 维状态
    输出：4 个动作各自的 Q 值

    结构：三层全连接，ReLU 激活。
    隐藏层 256 个神经元，比 CartPole 大，因为 LunarLander 状态更复杂。
    """
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


# ─────────────────────────────────────────────
# 经验回放缓冲区
# ─────────────────────────────────────────────
class ReplayBuffer:
    """
    存储历史 (s, a, r, s', done) 五元组。
    训练时从中随机采样，打破时序相关性，让样本近似 i.i.d.
    使用 deque(maxlen) 实现固定容量，超出时自动丢弃最旧的数据。
    """
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


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def train():
    env     = gym.make("LunarLander-v3")
    obs_dim = env.observation_space.shape[0]   # 8
    act_dim = env.action_space.n               # 4

    # 两个网络：policy_net 持续更新，target_net 定期同步
    policy_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # target_net 只用来推理，不计算梯度

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay    = ReplayBuffer(REPLAY_SIZE)
    epsilon   = EPSILON_START
    rewards   = []

    for ep in range(EPISODES):
        obs, _       = env.reset()
        total_reward = 0

        while True:
            # ── 动作选择：ε-greedy ──────────────────────
            # 以 ε 的概率随机探索，以 1-ε 的概率贪心利用
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    t      = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = int(policy_net(t).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 将这条经验存入回放池
            replay.push(obs, action, reward, next_obs, float(done))
            obs          = next_obs
            total_reward += reward

            # ── 网络更新 ────────────────────────────────
            if len(replay) >= MIN_REPLAY:
                s, a, r, s_, d = replay.sample(BATCH_SIZE)

                # 当前网络对已执行动作的 Q 值估计
                # gather(1, a) 取出每个样本实际执行动作对应的 Q 值
                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                # TD 目标：用目标网络计算下一状态的最大 Q 值
                # 注意：这里"选择动作"和"评估价值"用的是同一个网络（target_net）
                # → 这正是 Q 值高估的根源，Double DQN 会拆分这两步
                with torch.no_grad():
                    max_next_q = target_net(s_).max(dim=1).values
                    # done=1 时未来价值为 0（episode 结束）
                    td_target  = r + GAMMA * max_next_q * (1.0 - d)

                # 均方误差损失
                loss = nn.functional.mse_loss(q_values, td_target)

                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：防止梯度爆炸，稳定训练
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

            if done:
                break

        # ε 衰减：随着训练推进，逐步减少随机探索
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards.append(total_reward)

        # 定期同步目标网络
        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 每 50 回合打印一次进度
        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards[-50:])
            # >= 200 视为"解决"
            solved = "✓ 已解决" if avg >= 200 else ""
            print(f"Episode {ep+1:4d} | avg(50): {avg:7.1f} | ε={epsilon:.3f} {solved}")

    env.close()
    return policy_net, rewards


# ─────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────
def evaluate(policy_net: QNetwork, episodes: int = 20):
    """纯贪心策略跑评估，不做任何随机探索。"""
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
    print("=== LunarLander DQN ===")
    print("【注意】标准 DQN 存在 Q 值高估问题，训练后期可能不稳定")
    print("       Double DQN（double_dqn.py）可改善此问题\n")
    policy_net, rewards = train()
    print("\n=== 评估 ===")
    evaluate(policy_net)
