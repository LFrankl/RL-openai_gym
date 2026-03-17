"""
BipedalWalker-v3 — TD3 (Twin Delayed Deep Deterministic Policy Gradient)
=========================================================================

【TD3 vs SAC 的核心差异】

  SAC 学习的是随机策略 π(a|s)，动作从分布中采样，天然带有探索性。
  TD3 学习的是确定性策略 μ(s)：给定状态，策略直接输出一个确定动作。

  确定性策略的好处：
    - 梯度计算直接，不需要重参数化技巧
    - 推导自 Deterministic Policy Gradient（DPG，Silver et al., 2014）

  确定性策略的问题：
    - 没有内在随机性，需要外部噪声来探索
    - 容易高估 Q 值（确定性策略下 TD 目标偏差更大）

【TD3 的三个核心改进（相比 DDPG）】

  1. 双 Critic（Twin Q-Networks）
     与 SAC 类似，用两个独立的 Q 网络取 min 作为 TD 目标：
       y = r + γ(1-done) · min(Q1', Q2')(s', ã')
     其中 ã' 是加了目标策略噪声的下一步动作（见改进3）。

  2. 延迟策略更新（Delayed Policy Updates）
     Critic 每步更新，但 Actor 每隔 d 步才更新一次（d=2 默认）。
     直觉：Critic 需要先收敛到准确估计，再用它来指导 Actor。
     如果 Actor 和 Critic 同频更新，Critic 还没准确时 Actor 就跑偏了。

  3. 目标策略平滑（Target Policy Smoothing）
     TD 目标中的 Q'(s', ã') 对 ã' 很敏感，若 Q' 有尖峰，
     确定性目标动作 μ'(s') 容易落在尖峰上导致过估计。
     TD3 在目标动作上加噪声并截断，平滑 Q' 的估计：
       ã' = clip(μ'(s') + clip(ε, -c, c),  a_low, a_high),  ε ~ N(0, σ)
     本质是用 Q' 在目标动作邻域上的期望代替点估计，减少方差。

【探索策略】

  训练时在确定性动作上叠加 Ornstein-Uhlenbeck（OU）噪声或高斯噪声：
    a_explore = clip(μ(s) + ε,  -1, 1),  ε ~ N(0, σ_explore)
  评估时直接用 μ(s)，不加噪声。
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
EPISODES           = 1_500
MAX_STEPS          = 1_600
BUFFER_SIZE        = 300_000
BATCH_SIZE         = 256
GAMMA              = 0.99
TAU                = 0.005         # 目标网络软更新系数
LR_ACTOR           = 3e-4
LR_CRITIC          = 3e-4
POLICY_DELAY       = 2             # Actor 更新频率（每 d 次 Critic 更新）
EXPLORE_NOISE      = 0.1           # 训练时探索噪声标准差
TARGET_NOISE       = 0.2           # 目标策略平滑噪声标准差
TARGET_NOISE_CLIP  = 0.5           # 目标噪声截断范围
WARMUP_STEPS       = 5_000
HIDDEN_DIM         = 256

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


# ── Actor：确定性策略 ──────────────────────────────────────────────────────
class Actor(nn.Module):
    """
    直接输出确定性动作 a = tanh(μ(s))，映射到 [-1, 1]^act_dim。
    没有方差项，不采样，梯度直接通过网络反传。
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, act_dim), nn.Tanh(),  # 输出直接在 [-1, 1]
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


# ── Critic：双 Q 网络 ──────────────────────────────────────────────────────
class Critic(nn.Module):
    """两个独立的 Q(s,a) 网络，取 min 消除过估计。"""

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

    def q1_only(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Actor 更新时只用 Q1 的梯度，避免两个 Critic 梯度相互干扰。"""
        return self.q1(torch.cat([s, a], dim=-1))


# ── TD3 Agent ──────────────────────────────────────────────────────────────
class TD3Agent:
    def __init__(self, obs_dim: int, act_dim: int):
        self.act_dim = act_dim

        self.actor  = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic = Critic(obs_dim, act_dim).to(DEVICE)

        # 目标网络（软更新，不直接训练）
        self.actor_target  = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic_target = Critic(obs_dim, act_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in (*self.actor_target.parameters(), *self.critic_target.parameters()):
            p.requires_grad = False

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer      = ReplayBuffer(BUFFER_SIZE)
        self.update_step = 0   # 记录更新次数，用于延迟策略更新

    def select_action(self, s: np.ndarray, add_noise: bool = True) -> np.ndarray:
        s_t = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a = self.actor(s_t).squeeze(0).cpu().numpy()
        if add_noise:
            noise = np.random.normal(0, EXPLORE_NOISE, size=a.shape)
            a = np.clip(a + noise, -1.0, 1.0)
        return a

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        s, a, r, s_next, done = self.buffer.sample(BATCH_SIZE)
        self.update_step += 1

        # ── 更新 Critic ──────────────────────────────────────────────────
        with torch.no_grad():
            # 目标策略平滑：在目标动作上加截断高斯噪声
            noise = torch.randn_like(a) * TARGET_NOISE
            noise = noise.clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
            a_next = (self.actor_target(s_next) + noise).clamp(-1.0, 1.0)

            # TD 目标：用目标网络的 min Q
            q1_t, q2_t = self.critic_target(s_next, a_next)
            q_target = r + GAMMA * (1 - done) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(s, a)
        loss_critic = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        # ── 延迟策略更新：每 d 步才更新 Actor 和目标网络 ─────────────────
        if self.update_step % POLICY_DELAY == 0:
            # Actor 目标：最大化 Q1(s, μ(s))
            loss_actor = -self.critic.q1_only(s, self.actor(s)).mean()

            self.opt_actor.zero_grad()
            loss_actor.backward()
            self.opt_actor.step()

            # 软更新目标网络
            for p, p_tgt in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_tgt.data.mul_(1 - TAU).add_(TAU * p.data)
            for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_tgt.data.mul_(1 - TAU).add_(TAU * p.data)


# ── 训练 ───────────────────────────────────────────────────────────────────
def train():
    env = gym.make("BipedalWalker-v3")
    obs_dim = env.observation_space.shape[0]   # 24
    act_dim = env.action_space.shape[0]         # 4

    agent = TD3Agent(obs_dim, act_dim)
    total_steps = 0
    best_avg    = -float("inf")

    print(f"TD3 | obs={obs_dim} act={act_dim} | device={DEVICE}")
    print(f"{'Ep':>6} {'Score':>8} {'Avg100':>8} {'Steps':>8}")
    print("-" * 40)

    scores = []
    for ep in range(1, EPISODES + 1):
        s, _ = env.reset()
        ep_score = 0.0

        for _ in range(MAX_STEPS):
            if total_steps < WARMUP_STEPS:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s, add_noise=True)

            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            agent.buffer.push(s, a, r, s_next, float(terminated))

            if total_steps >= WARMUP_STEPS:
                agent.update()

            s           = s_next
            ep_score   += r
            total_steps += 1

            if done:
                break

        scores.append(ep_score)
        avg100 = np.mean(scores[-100:])

        if ep % 10 == 0:
            print(f"{ep:>6} {ep_score:>8.1f} {avg100:>8.1f} {total_steps:>8}")

        if avg100 > best_avg and len(scores) >= 100:
            best_avg = avg100
            torch.save(agent.actor.state_dict(), "bipedalwalker/td3_best.pth")

    env.close()
    return agent, scores


# ── 评估 ───────────────────────────────────────────────────────────────────
def evaluate(agent: TD3Agent, episodes: int = 10, render: bool = True):
    env = gym.make("BipedalWalker-v3", render_mode="human" if render else None)
    scores = []
    for ep in range(episodes):
        s, _ = env.reset()
        total, steps = 0.0, 0
        while True:
            a = agent.select_action(s, add_noise=False)
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
