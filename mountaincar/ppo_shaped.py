"""
MountainCar-v0 — PPO + 直接奖励重设计
--------------------------------------
问题根因分析：
  标准奖励 r = -1（每步）在 MountainCar 是高度稀疏的，
  差分势函数 γΦ(s') - Φ(s) 在量级上远小于累计 -200，
  策略梯度无法从中提取有效学习信号。

解决方案：直接重设计奖励（非势函数差分）
  r_custom = position + 0.5       # 位置越靠右越好，范围 [-0.7, 1.1]
           + 3 * |velocity|       # 速度越大越好，范围 [0, 0.21]
           + 100 * terminated     # 到达山顶给巨大奖励

注意：这种方式不保留原始奖励的最优策略，仅用于教学展示。
     如需保最优策略，须严格遵守差分势函数形式（Ng et al., 1999）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym


ROLLOUT_STEPS = 2048
PPO_EPOCHS    = 10
MINI_BATCH    = 64
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
VF_COEF       = 0.5
ENT_COEF      = 0.01
ALPHA         = 3e-4
MAX_GRAD_NORM = 0.5
MAX_STEPS     = 300_000

DEVICE = torch.device("cpu")
# DEVICE = torch.device("mps")  # Mac M4 启用 GPU 加速（取消注释即可）


def custom_reward(obs, next_obs, terminated):
    pos, vel   = next_obs
    r_pos      = pos + 0.5          # 位置奖励
    r_vel      = 3.0 * abs(vel)     # 速度奖励
    r_terminal = 100.0 * terminated  # 到顶奖励
    return r_pos + r_vel + r_terminal


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor  = nn.Linear(64, 3)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.actor(f), self.critic(f).squeeze(-1)

    def get_action(self, obs):
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, value = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs, actions):
        logits, values = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


def compute_gae(rewards, values, dones, last_value):
    n   = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    ext = values + [last_value]
    for t in reversed(range(n)):
        delta = rewards[t] + GAMMA * ext[t+1] * (1 - dones[t]) - ext[t]
        gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        adv[t] = gae
    return adv, adv + np.array(values, dtype=np.float32)


def train():
    env   = gym.make("MountainCar-v0")
    model = ActorCritic().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=ALPHA, eps=1e-5)

    buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done = [], [], [], [], [], []
    obs, _      = env.reset()
    ep_reward   = 0   # 记录原始奖励
    ep_rewards  = []
    successes   = []
    total_steps = 0
    last_log    = 0

    while total_steps < MAX_STEPS:
        for _ in range(ROLLOUT_STEPS):
            action, logp, value = model.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            r = custom_reward(obs, next_obs, terminated)

            buf_obs.append(obs);  buf_act.append(action)
            buf_logp.append(logp); buf_val.append(value)
            buf_rew.append(r);    buf_done.append(float(done))

            ep_reward   += reward   # 记录原始 -1 奖励
            total_steps += 1
            obs          = next_obs

            if done:
                ep_rewards.append(ep_reward)
                successes.append(int(terminated))
                ep_reward = 0
                obs, _    = env.reset()

        with torch.no_grad():
            t         = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, last_v = model(t)
            last_val  = last_v.item() * (1 - buf_done[-1])

        adv, ret = compute_gae(buf_rew, buf_val, buf_done, last_val)

        t_obs  = torch.tensor(np.array(buf_obs),  dtype=torch.float32, device=DEVICE)
        t_act  = torch.tensor(buf_act,             dtype=torch.long,    device=DEVICE)
        t_logp = torch.tensor(buf_logp,            dtype=torch.float32, device=DEVICE)
        t_adv  = torch.tensor(adv,                 dtype=torch.float32, device=DEVICE)
        t_ret  = torch.tensor(ret,                 dtype=torch.float32, device=DEVICE)
        t_adv  = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(ROLLOUT_STEPS, device=DEVICE)
            for start in range(0, ROLLOUT_STEPS, MINI_BATCH):
                mb = idx[start:start+MINI_BATCH]
                new_logp, entropy, new_val = model.evaluate(t_obs[mb], t_act[mb])
                ratio  = (new_logp - t_logp[mb]).exp()
                adv_mb = t_adv[mb]
                loss = (
                    -torch.min(ratio * adv_mb,
                               ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS) * adv_mb).mean()
                    + VF_COEF * nn.functional.mse_loss(new_val, t_ret[mb])
                    - ENT_COEF * entropy.mean()
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()

        buf_obs.clear(); buf_act.clear(); buf_logp.clear()
        buf_val.clear(); buf_rew.clear(); buf_done.clear()

        n = len(ep_rewards)
        if n - last_log >= 50 and n > 0:
            avg      = np.mean(ep_rewards[-50:])
            suc_rate = np.mean(successes[-50:]) * 100
            print(f"Episode {n:4d} | avg(原始): {avg:7.1f} | 到顶率: {suc_rate:5.1f}% | steps: {total_steps}")
            last_log = n

    env.close()
    return model, ep_rewards


def evaluate(model, episodes=20):
    env   = gym.make("MountainCar-v0")
    total = 0
    wins  = 0
    model.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _    = env.reset()
            ep_reward = 0
            while True:
                t      = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = int(Categorical(logits=model(t)[0]).probs.argmax().item())
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    if terminated:
                        wins += 1
                    break
            total += ep_reward
    env.close()
    print(f"\n评估 {episodes} 回合 → 平均得分(原始): {total/episodes:.1f} | 到顶率: {wins/episodes*100:.0f}%")


if __name__ == "__main__":
    print("=== PPO + 自定义奖励 ===\n")
    model, rewards = train()
    evaluate(model)
