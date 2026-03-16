"""
MountainCar-v0 — Q-Learning（原版，展示稀疏奖励失败）
------------------------------------------------------
MountainCar 的奖励结构：
  - 每步 -1（鼓励尽快到达）
  - 到达山顶（position >= 0.5）终止，episode 结束
  - 最多 200 步，到不了山顶得 -200 分

状态: [position ∈ (-1.2, 0.6), velocity ∈ (-0.07, 0.07)]
动作: 0=向左推, 1=不动, 2=向右推

难点：小车动力不足，必须左右摇摆积累动能才能上山。
     随机探索极难碰到山顶，Q-Table 几乎学不到有效信号。
"""

import numpy as np
import gymnasium as gym


# ── 超参数 ─────────────────────────────────────────────
EPISODES      = 5_000
ALPHA         = 0.1
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.01
EPSILON_DECAY = 0.995

BUCKETS = (20, 20)   # position × velocity

OBS_LOW  = np.array([-1.2, -0.07])
OBS_HIGH = np.array([ 0.6,  0.07])


def discretize(obs):
    clipped = np.clip(obs, OBS_LOW, OBS_HIGH)
    ratios  = (clipped - OBS_LOW) / (OBS_HIGH - OBS_LOW)
    indices = (ratios * np.array(BUCKETS)).astype(int)
    return tuple(np.clip(indices, 0, np.array(BUCKETS) - 1))


def train():
    env     = gym.make("MountainCar-v0")
    q_table = np.zeros(BUCKETS + (env.action_space.n,))
    epsilon = EPSILON_START
    rewards = []
    success = 0

    for ep in range(EPISODES):
        obs, _       = env.reset()
        state        = discretize(obs)
        total_reward = 0

        while True:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs)
            done = terminated or truncated

            best_next = np.max(q_table[next_state])
            q_table[state][action] += ALPHA * (
                reward + GAMMA * best_next - q_table[state][action]
            )

            state        = next_state
            total_reward += reward
            if done:
                break

        if total_reward > -200:
            success += 1

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            avg = np.mean(rewards[-500:])
            print(f"Episode {ep+1:5d} | avg: {avg:7.1f} | success(last 500): {success} | ε={epsilon:.3f}")
            success = 0

    env.close()
    return q_table, rewards


if __name__ == "__main__":
    print("=== Q-Learning（无奖励塑形）===")
    print("预期结果：几乎学不会，avg 接近 -200\n")
    train()
