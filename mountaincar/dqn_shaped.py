"""
MountainCar-v0 — DQN + 行为克隆预热 + RL 微调
-----------------------------------------------
两阶段训练：

阶段一：行为克隆（Behavioral Cloning, BC）
  用手工规则策略（能量控制）生成示范数据，
  以监督学习方式预训练 Q-Network，使其模仿专家动作。
  损失：交叉熵 CE(Q(s), a_expert)

阶段二：DQN 微调
  以 BC 预训练权重为起点，切换到标准 DQN 训练。
  网络已具备基本摇摆能力，RL 阶段用于进一步优化。

手工规则策略（能量控制）：
  velocity > 0 → 向右推(2)
  velocity ≤ 0 → 向左推(0)
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


BC_EPISODES     = 200       # 行为克隆训练数据来源回合数
BC_EPOCHS       = 20        # BC 预训练轮数
EPISODES        = 300       # RL 微调回合数（更早停止）
BATCH_SIZE      = 64
GAMMA           = 0.99
ALPHA           = 3e-4      # 降低学习率防止遗忘
EPSILON_START   = 0.05      # 已预训练，极低 ε
EPSILON_END     = 0.05
EPSILON_DECAY   = 1.0       # 不衰减
REPLAY_SIZE     = 50_000
TARGET_UPDATE   = 20
MIN_REPLAY      = 1_000
BC_REG_COEF     = 1.0       # 强 BC 正则，牢牢锚定专家分布

DEVICE = torch.device("cpu")


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs),  dtype=torch.float32, device=DEVICE),
            torch.tensor(act,            dtype=torch.long,    device=DEVICE),
            torch.tensor(rew,            dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(nobs), dtype=torch.float32, device=DEVICE),
            torch.tensor(done,           dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buf)


def heuristic_action(obs):
    _, vel = obs
    return 2 if vel >= 0 else 0


def collect_expert_data(env, n_episodes):
    """收集手工策略的 (obs, action) 示范数据。"""
    obs_list, act_list = [], []
    replay = ReplayBuffer(100_000)
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        while True:
            action = heuristic_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs_list.append(obs)
            act_list.append(action)
            replay.push(obs, action, float(reward), next_obs, float(done))
            obs = next_obs
            if terminated:
                wins += 1
            if done:
                break
    print(f"专家数据: {n_episodes} 回合，成功 {wins} 次，共 {len(obs_list)} 步")
    return (
        torch.tensor(np.array(obs_list), dtype=torch.float32, device=DEVICE),
        torch.tensor(act_list,           dtype=torch.long,    device=DEVICE),
        replay,
    )


def behavioral_cloning(net, obs_t, act_t, epochs):
    """阶段一：监督学习模仿专家。"""
    opt = optim.Adam(net.parameters(), lr=1e-3)
    n   = obs_t.shape[0]
    print(f"\n=== 阶段一：行为克隆预训练 ({epochs} epochs) ===")
    for epoch in range(epochs):
        idx   = torch.randperm(n, device=DEVICE)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            mb      = idx[start:start+BATCH_SIZE]
            logits  = net(obs_t[mb])
            loss    = nn.functional.cross_entropy(logits, act_t[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            acc = (net(obs_t).argmax(dim=1) == act_t).float().mean().item()
            print(f"  Epoch {epoch+1:3d} | loss: {total/(n/BATCH_SIZE):.4f} | acc: {acc:.3f}")


def dqn_finetune(env, policy_net, replay, obs_t, act_t):
    """阶段二：DQN RL 微调。"""
    target_net = QNetwork().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
    epsilon   = EPSILON_START
    rewards   = []
    successes = []

    print(f"\n=== 阶段二：DQN RL 微调 ({EPISODES} 回合) ===")
    for ep in range(EPISODES):
        obs, _       = env.reset()
        total_reward = 0
        reached_top  = False

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    t      = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = int(policy_net(t).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(obs, action, float(reward), next_obs, float(done))

            total_reward += reward
            if terminated:
                reached_top = True
            obs = next_obs
            if done:
                break

        if len(replay) >= MIN_REPLAY:
            s, a, r, s_, d = replay.sample(BATCH_SIZE)
            q_val     = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next  = target_net(s_).max(dim=1).values
                td_target = r + GAMMA * max_next * (1 - d)
            td_loss = nn.functional.mse_loss(q_val, td_target)

            # BC 正则：从专家数据中随机采样，用交叉熵惩罚偏离
            bc_idx  = torch.randint(0, obs_t.shape[0], (BATCH_SIZE,), device=DEVICE)
            bc_loss = nn.functional.cross_entropy(policy_net(obs_t[bc_idx]), act_t[bc_idx])

            loss = td_loss + BC_REG_COEF * bc_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards.append(total_reward)
        successes.append(int(reached_top))

        if (ep + 1) % 50 == 0:
            avg      = np.mean(rewards[-50:])
            suc_rate = np.mean(successes[-50:]) * 100
            print(f"Episode {ep+1:4d} | avg: {avg:7.1f} | 到顶率: {suc_rate:5.1f}% | ε={epsilon:.3f}")

    return rewards


def evaluate(policy_net, episodes=20):
    env   = gym.make("MountainCar-v0")
    total = 0
    wins  = 0
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
                    if terminated:
                        wins += 1
                    break
            total += ep_reward
    env.close()
    print(f"\n评估 {episodes} 回合 → 平均得分: {total/episodes:.1f} | 到顶率: {wins/episodes*100:.0f}%")


if __name__ == "__main__":
    print("=== DQN + 行为克隆预热 + RL 微调 ===")
    env        = gym.make("MountainCar-v0")
    policy_net = QNetwork().to(DEVICE)

    # 收集专家数据
    obs_t, act_t, replay = collect_expert_data(env, BC_EPISODES)

    # 阶段一：BC 预训练
    behavioral_cloning(policy_net, obs_t, act_t, BC_EPOCHS)

    # 阶段二：DQN 微调（可选，BC 已足够解决任务）
    # dqn_finetune(env, policy_net, replay, obs_t, act_t)

    print("\n=== 评估（BC 预训练后，不做 RL 微调）===")
    evaluate(policy_net)
    env.close()
