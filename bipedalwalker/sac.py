"""
BipedalWalker-v3 — SAC (Soft Actor-Critic)
===========================================

【为什么离散动作算法在这里失效？】

  DQN 的核心操作是 argmax_a Q(s, a)。
  BipedalWalker 的动作空间是 [-1, 1]^4（4个关节的连续力矩），
  连续空间里无法穷举所有动作来取最大值——这个 argmax 没有解析解。

  因此必须切换到能直接处理连续动作的算法。

【SAC 的核心思想：最大熵强化学习】

  标准 RL 目标：最大化期望累计回报
    J(π) = E[ Σ γ^t r_t ]

  SAC 在每一步额外奖励策略的"熵"，鼓励保持探索性：
    J_SAC(π) = E[ Σ γ^t (r_t + α · H(π(·|s_t))) ]

  其中 H(π(·|s)) = -E_a[log π(a|s)] 是策略在状态 s 下的香农熵。
  温度参数 α 控制熵的权重——α 越大，越鼓励探索；α=0 退化为标准 RL。

  最大熵框架的好处：
    1. 自动探索：不需要手动设计探索策略（ε-greedy 等）
    2. 鲁棒性：多种近似最优策略并存，面对扰动更鲁棒
    3. 避免过早收敛到次优确定性策略

【SAC 的三个核心组件】

  1. 双 Critic（Twin Q-Networks）
     训练两个独立的 Q 网络，取 min 值作为 TD 目标，消除 Q 值高估：
       y = r + γ(1-done) · [min(Q1', Q2')(s', ã') - α log π(ã'|s')]
     其中 ã' ~ π(·|s') 是从当前策略采样的下一步动作。

  2. 重参数化采样（Reparameterization Trick）
     直接对 log π 求梯度有高方差。SAC 通过以下技巧低方差地更新 Actor：
       a = tanh(μ(s) + σ(s) · ε),   ε ~ N(0, I)
     这让采样操作变成确定性函数，梯度可以直接通过 tanh 反传。

  3. 自动温度调节（Automatic Entropy Tuning）
     手动调 α 麻烦，SAC 将 α 也作为可学习参数，
     通过最小化以下损失自动调节：
       L(α) = -α · E_a[log π(a|s) + target_entropy]
     target_entropy 设为 -|A|（动作维度的负数），是理论推荐值。
     直觉：如果策略熵低于目标，就增大 α 鼓励探索；反之减小 α。

【训练流程】

  收集经验 → 存入 Replay Buffer
  采样 mini-batch：
    更新 Critic：最小化 TD 误差（两个独立）
    更新 Actor：最大化 min(Q1, Q2) - α log π（重参数化）
    更新 α：使策略熵趋近 target_entropy
    软更新目标网络：θ' ← τθ + (1-τ)θ'
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

# ── 超参数 ─────────────────────────────────────────────────────────────────
EPISODES        = 1_500
MAX_STEPS       = 1_600        # BipedalWalker 每回合最多 1600 步
BUFFER_SIZE     = 300_000
BATCH_SIZE      = 256
GAMMA           = 0.99
TAU             = 0.005        # 目标网络软更新系数
LR_ACTOR        = 3e-4
LR_CRITIC       = 3e-4
LR_ALPHA        = 3e-4
WARMUP_STEPS    = 5_000        # 前 N 步随机探索，不更新网络
UPDATE_EVERY    = 1            # 每采集多少步更新一次
HIDDEN_DIM      = 256

# 目标熵：-|A|，动作维度为 4
TARGET_ENTROPY  = -4.0

DEVICE = torch.device("cpu")
# DEVICE = torch.device("mps")  # Mac M4 启用 GPU 加速（取消注释即可）


# ── Replay Buffer ──────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(DEVICE),
            torch.FloatTensor(np.array(a)).to(DEVICE),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(s_next)).to(DEVICE),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ── Actor 网络 ─────────────────────────────────────────────────────────────
class Actor(nn.Module):
    """
    输出连续动作的高斯分布参数 (μ, log σ)，
    通过重参数化 a = tanh(μ + σ·ε) 采样并计算对数概率。
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        )
        self.mu_head      = nn.Linear(HIDDEN_DIM, act_dim)
        self.log_std_head = nn.Linear(HIDDEN_DIM, act_dim)

    def forward(self, s: torch.Tensor):
        feat    = self.net(s)
        mu      = self.mu_head(feat)
        log_std = self.log_std_head(feat).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, s: torch.Tensor):
        """
        重参数化采样：a = tanh(μ + σ·ε)，ε ~ N(0,I)
        同时计算修正后的对数概率（tanh 变换的 Jacobian 修正）：
          log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh(u)^2)
        """
        mu, log_std = self.forward(s)
        std = log_std.exp()
        # 重参数化：ε ~ N(0,I)，u = μ + σ·ε
        eps = torch.randn_like(mu)
        u   = mu + std * eps
        a   = torch.tanh(u)

        # 修正对数概率（tanh 压缩了概率密度）
        log_prob = (
            torch.distributions.Normal(mu, std).log_prob(u)
            - torch.log(1 - a.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)

        return a, log_prob

    def deterministic_action(self, s: torch.Tensor):
        """评估时用均值，不采样，更稳定。"""
        mu, _ = self.forward(s)
        return torch.tanh(mu)


# ── Critic 网络（Q 函数）────────────────────────────────────────────────────
class Critic(nn.Module):
    """
    双 Critic：两个独立的 Q 网络，输入 (s, a)，输出标量 Q 值。
    取 min(Q1, Q2) 作为 TD 目标，消除过估计。
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        def make_q():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, HIDDEN_DIM), nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                nn.Linear(HIDDEN_DIM, 1),
            )
        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)


# ── SAC Agent ──────────────────────────────────────────────────────────────
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int):
        self.actor  = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic = Critic(obs_dim, act_dim).to(DEVICE)

        # 目标 Critic（软更新，不直接训练）
        self.critic_target = Critic(obs_dim, act_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # 可学习的温度参数 log(α)（取 log 确保 α > 0）
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha     = self.log_alpha.exp().item()

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.opt_alpha  = optim.Adam([self.log_alpha],         lr=LR_ALPHA)

        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, s: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s_t = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if deterministic:
                a = self.actor.deterministic_action(s_t)
            else:
                a, _ = self.actor.sample(s_t)
        return a.squeeze(0).cpu().numpy()

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        s, a, r, s_next, done = self.buffer.sample(BATCH_SIZE)

        # ── 更新 Critic ──────────────────────────────────────────────────
        with torch.no_grad():
            a_next, log_prob_next = self.actor.sample(s_next)
            # TD 目标：min Q' - α log π（带熵的 Bellman 目标）
            q_target = r + GAMMA * (1 - done) * (
                self.critic_target.q_min(s_next, a_next) - self.alpha * log_prob_next
            )

        q1, q2 = self.critic(s, a)
        loss_critic = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        # ── 更新 Actor ───────────────────────────────────────────────────
        a_curr, log_prob_curr = self.actor.sample(s)
        # Actor 目标：最大化 E[min Q - α log π]
        loss_actor = (self.alpha * log_prob_curr - self.critic.q_min(s, a_curr)).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        # ── 更新温度 α ───────────────────────────────────────────────────
        # 目标：使熵接近 target_entropy
        # L(α) = -α · E[log π + target_entropy]
        loss_alpha = -(self.log_alpha * (log_prob_curr + TARGET_ENTROPY).detach()).mean()

        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()
        self.alpha = self.log_alpha.exp().item()

        # ── 软更新目标 Critic ─────────────────────────────────────────────
        for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_tgt.data.mul_(1 - TAU).add_(TAU * p.data)


# ── 训练 ───────────────────────────────────────────────────────────────────
def train():
    env = gym.make("BipedalWalker-v3")
    obs_dim = env.observation_space.shape[0]   # 24
    act_dim = env.action_space.shape[0]         # 4

    agent = SACAgent(obs_dim, act_dim)
    total_steps = 0
    best_avg    = -float("inf")

    print(f"SAC | obs={obs_dim} act={act_dim} | device={DEVICE}")
    print(f"{'Ep':>6} {'Score':>8} {'Avg100':>8} {'Alpha':>7} {'Steps':>8}")
    print("-" * 50)

    scores = []
    for ep in range(1, EPISODES + 1):
        s, _ = env.reset()
        ep_score = 0.0

        for _ in range(MAX_STEPS):
            # 热身阶段随机探索，避免 Replay Buffer 全是差样本
            if total_steps < WARMUP_STEPS:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s)

            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            agent.buffer.push(s, a, r, s_next, float(terminated))

            if total_steps >= WARMUP_STEPS and total_steps % UPDATE_EVERY == 0:
                agent.update()

            s          = s_next
            ep_score  += r
            total_steps += 1

            if done:
                break

        scores.append(ep_score)
        avg100 = np.mean(scores[-100:])

        if ep % 10 == 0:
            print(f"{ep:>6} {ep_score:>8.1f} {avg100:>8.1f} {agent.alpha:>7.4f} {total_steps:>8}")

        if avg100 > best_avg and len(scores) >= 100:
            best_avg = avg100
            torch.save(agent.actor.state_dict(), "bipedalwalker/sac_best.pth")

    env.close()
    return agent, scores


# ── 评估 ───────────────────────────────────────────────────────────────────
def evaluate(agent: SACAgent, episodes: int = 10, render: bool = True):
    env = gym.make("BipedalWalker-v3", render_mode="human" if render else None)
    scores = []
    for ep in range(episodes):
        s, _ = env.reset()
        total, steps = 0.0, 0
        while True:
            a = agent.select_action(s, deterministic=True)
            s, r, terminated, truncated, _ = env.step(a)
            total += r
            steps += 1
            if terminated or truncated:
                break
        scores.append(total)
        print(f"  评估第 {ep+1} 回合: {total:.1f}  ({steps} 步)")
    env.close()
    print(f"\n  均分: {np.mean(scores):.1f}  |  最高: {max(scores):.1f}")
    return scores


# ── 主程序 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent, scores = train()
    print("\n训练完成，开始评估（渲染动画）...")
    evaluate(agent)
