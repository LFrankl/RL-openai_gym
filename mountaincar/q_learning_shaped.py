"""
MountainCar-v0 — Q-Learning + 奖励塑形 + 乐观初始化
-----------------------------------------------------
纯随机探索（ε-greedy）在 MountainCar 几乎不可能碰到山顶：
  - 200 步内随机摇摆到山顶的概率极低
  - 没有正向信号，Q-Table 无从学习

两种改进方向：
  1. 乐观初始化（Optimistic Initialization）
     初始化 Q(s,a) = Q_0 > 0，使智能体倾向于探索从未访问的状态
     一旦访问后实际奖励 < Q_0，Q 值下降，转向其他未探索方向
     本质是系统性探索，而非随机探索

  2. 奖励塑形（Reward Shaping）
     Φ(s) = w_vel * |velocity|（只用速度，引导积累动能）
     位置项干扰较大（让小车总想往右，而非摇摆），去掉
"""

import numpy as np
import gymnasium as gym


EPISODES      = 3_000
ALPHA         = 0.3
GAMMA         = 0.99
EPSILON       = 0.1        # 固定小 ε，配合乐观初始化做少量随机探索
Q_INIT        = 5.0        # 乐观初始值

BUCKETS  = (20, 20)
OBS_LOW  = np.array([-1.2, -0.07])
OBS_HIGH = np.array([ 0.6,  0.07])

W_VELOCITY = 40.0          # 只用速度势函数，去掉位置项


def discretize(obs):
    clipped = np.clip(obs, OBS_LOW, OBS_HIGH)
    ratios  = (clipped - OBS_LOW) / (OBS_HIGH - OBS_LOW)
    indices = (ratios * np.array(BUCKETS)).astype(int)
    return tuple(np.clip(indices, 0, np.array(BUCKETS) - 1))


def potential(obs):
    return W_VELOCITY * abs(obs[1])


def train():
    env     = gym.make("MountainCar-v0")
    # 乐观初始化：Q 值全部设为正数
    q_table = np.full(BUCKETS + (env.action_space.n,), Q_INIT)
    rewards   = []
    successes = []

    for ep in range(EPISODES):
        obs, _       = env.reset()
        state        = discretize(obs)
        total_reward = 0
        reached_top  = False

        while True:
            if np.random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs)
            done = terminated or truncated

            shaped_reward = reward + GAMMA * potential(next_obs) - potential(obs)

            best_next = np.max(q_table[next_state])
            q_table[state][action] += ALPHA * (
                shaped_reward + GAMMA * best_next - q_table[state][action]
            )

            state        = next_state
            total_reward += reward
            if terminated:
                reached_top = True
            if done:
                break

        rewards.append(total_reward)
        successes.append(int(reached_top))

        if (ep + 1) % 300 == 0:
            avg      = np.mean(rewards[-300:])
            suc_rate = np.mean(successes[-300:]) * 100
            print(f"Episode {ep+1:4d} | avg: {avg:7.1f} | 到顶率: {suc_rate:5.1f}%")

    env.close()
    return q_table, rewards


def evaluate(q_table, episodes=20):
    env   = gym.make("MountainCar-v0")
    total = 0
    wins  = 0
    for _ in range(episodes):
        obs, _    = env.reset()
        ep_reward = 0
        while True:
            state  = discretize(obs)
            action = int(np.argmax(q_table[state]))
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
    print("=== Q-Learning + 奖励塑形 + 乐观初始化 ===\n")
    q_table, rewards = train()
    evaluate(q_table)
