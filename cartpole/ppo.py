"""
CartPole-v1 — PPO (Proximal Policy Optimization)
-------------------------------------------------
PPO 是策略梯度算法，直接优化策略 π(a|s)，不再维护 Q 值。
相比 DQN 更稳定，工业界主流（ChatGPT 的 RLHF 也用它）。

核心思路：
  1. Actor-Critic 网络    — Actor 输出动作概率，Critic 估计状态价值
  2. GAE                  — 广义优势估计，平衡偏差和方差
  3. Clipped Surrogate    — 限制策略更新幅度，防止"一步走太大"
     L_CLIP = E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) ]
  4. 多轮 epoch 复用数据  — 同一批轨迹数据更新 K 次，提升样本效率

训练流程：
  收集 T 步轨迹 → 计算 GAE 优势 → 多 epoch 更新网络 → 清空缓冲区 → 重复
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym


# ── 超参数 ─────────────────────────────────────────────
EPISODES        = 1_000
ROLLOUT_STEPS   = 2048      # 每次收集多少步轨迹再更新
PPO_EPOCHS      = 10        # 每批数据重复训练几轮
MINI_BATCH      = 64        # mini-batch 大小
GAMMA           = 0.99      # 折扣因子
GAE_LAMBDA      = 0.95      # GAE λ，越大越依赖远期奖励
CLIP_EPS        = 0.2       # PPO clip 范围
VF_COEF         = 0.5       # 价值函数损失系数
ENT_COEF        = 0.01      # 熵正则系数（鼓励探索）
ALPHA           = 3e-4      # Adam 学习率
MAX_GRAD_NORM   = 0.5       # 梯度裁剪

DEVICE = torch.device("cpu")
# DEVICE = torch.device("mps")  # Mac M4 启用 GPU 加速（取消注释即可）


# ── Actor-Critic 网络 ───────────────────────────────────
class ActorCritic(nn.Module):
    """共享主干，Actor 和 Critic 各接一个输出头。"""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(64, act_dim)   # 输出 logits
        self.critic = nn.Linear(64, 1)          # 输出状态价值 V(s)

    def forward(self, x: torch.Tensor):
        feat   = self.backbone(x)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value

    def get_action(self, obs: np.ndarray):
        """采样动作，返回 (action, log_prob, value)。"""
        x      = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, value = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """批量计算 log_prob、entropy、value，用于 PPO 更新。"""
        logits, values = self(obs)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, entropy, values


# ── GAE 计算 ────────────────────────────────────────────
def compute_gae(rewards, values, dones, last_value):
    """
    广义优势估计 (GAE-λ)
    advantages[t] = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    """
    n          = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae        = 0.0
    values_ext = values + [last_value]   # 在末尾补上 bootstrap value

    for t in reversed(range(n)):
        mask   = 1.0 - dones[t]
        delta  = rewards[t] + GAMMA * values_ext[t + 1] * mask - values_ext[t]
        gae    = delta + GAMMA * GAE_LAMBDA * mask * gae
        advantages[t] = gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ── 训练 ────────────────────────────────────────────────
def train():
    env   = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=ALPHA, eps=1e-5)

    # 轨迹缓冲区
    buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done = [], [], [], [], [], []

    obs, _       = env.reset()
    ep_reward    = 0
    ep_rewards   = []
    total_steps  = 0
    last_log     = 0

    while len(ep_rewards) < EPISODES:
        # ── 收集 ROLLOUT_STEPS 步 ──────────────────────
        for _ in range(ROLLOUT_STEPS):
            action, logp, value = model.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buf_obs.append(obs)
            buf_act.append(action)
            buf_logp.append(logp)
            buf_val.append(value)
            buf_rew.append(reward)
            buf_done.append(float(done))

            obs        = next_obs
            ep_reward += reward
            total_steps += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0
                obs, _   = env.reset()

        # bootstrap：如果最后一步没结束，用 V(s_T) 作为尾部估计
        with torch.no_grad():
            t          = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, last_v  = model(t)
            last_value = last_v.item() * (1.0 - buf_done[-1])

        advantages, returns = compute_gae(buf_rew, buf_val, buf_done, last_value)

        # 转 tensor
        t_obs  = torch.tensor(np.array(buf_obs),  dtype=torch.float32, device=DEVICE)
        t_act  = torch.tensor(buf_act,             dtype=torch.long,    device=DEVICE)
        t_logp = torch.tensor(buf_logp,            dtype=torch.float32, device=DEVICE)
        t_adv  = torch.tensor(advantages,          dtype=torch.float32, device=DEVICE)
        t_ret  = torch.tensor(returns,             dtype=torch.float32, device=DEVICE)

        # 优势标准化
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

        # ── PPO 多轮更新 ───────────────────────────────
        idx = torch.randperm(ROLLOUT_STEPS, device=DEVICE)
        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(ROLLOUT_STEPS, device=DEVICE)
            for start in range(0, ROLLOUT_STEPS, MINI_BATCH):
                mb = idx[start: start + MINI_BATCH]

                new_logp, entropy, new_val = model.evaluate(t_obs[mb], t_act[mb])

                # ratio = π_new / π_old
                ratio     = (new_logp - t_logp[mb]).exp()
                adv_mb    = t_adv[mb]

                # Clipped surrogate loss
                surr1     = ratio * adv_mb
                surr2     = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb
                actor_loss  = -torch.min(surr1, surr2).mean()

                # Value function loss
                critic_loss = nn.functional.mse_loss(new_val, t_ret[mb])

                # 熵 bonus（鼓励探索）
                entropy_loss = -entropy.mean()

                loss = actor_loss + VF_COEF * critic_loss + ENT_COEF * entropy_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()

        # 清空缓冲区
        buf_obs.clear(); buf_act.clear(); buf_logp.clear()
        buf_val.clear(); buf_rew.clear(); buf_done.clear()

        n = len(ep_rewards)
        if n - last_log >= 50:
            avg = np.mean(ep_rewards[-50:])
            print(f"Episode {n:4d} | avg(50): {avg:6.1f} | steps: {total_steps}")
            last_log = n

    env.close()
    return model, ep_rewards


# ── 评估 ────────────────────────────────────────────────
def evaluate(model: ActorCritic, episodes: int = 10):
    env   = gym.make("CartPole-v1")
    total = 0
    model.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _    = env.reset()
            ep_reward = 0
            while True:
                action, _, _ = model.get_action(obs)
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
    print("=== 训练 CartPole PPO ===")
    model, rewards = train()

    print("\n=== 评估 ===")
    evaluate(model)
