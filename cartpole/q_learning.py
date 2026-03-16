"""
CartPole-v1 — Q-Learning with discretized state space
------------------------------------------------------
CartPole 是经典控制问题：
  - 状态: 小车位置/速度 + 杆角度/角速度（4维连续）
  - 动作: 向左推(0) / 向右推(1)
  - 目标: 让杆保持直立，每步得 +1 奖励，最多 500 步

算法: Q-Learning (表格法)
  连续状态 → 离散化分桶 → Q-Table 更新
  Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
"""

import numpy as np
import gymnasium as gym


# ── 超参数 ─────────────────────────────────────────────
EPISODES      = 5_000   # 训练回合数
ALPHA         = 0.1     # 学习率
GAMMA         = 0.99    # 折扣因子
EPSILON_START = 1.0     # 初始探索率
EPSILON_END   = 0.01    # 最小探索率
EPSILON_DECAY = 0.995   # 每回合衰减

# 状态离散化: 每个维度分多少桶
BUCKETS = (6, 6, 12, 12)

# 观测空间裁剪上限（原始范围太大，手动收窄）
OBS_LOW  = np.array([-2.4, -2.0, -0.25, -3.5])
OBS_HIGH = np.array([ 2.4,  2.0,  0.25,  3.5])


def discretize(obs: np.ndarray) -> tuple:
    """将连续观测值映射到桶索引。"""
    clipped = np.clip(obs, OBS_LOW, OBS_HIGH)
    ratios  = (clipped - OBS_LOW) / (OBS_HIGH - OBS_LOW)   # [0, 1]
    indices = (ratios * np.array(BUCKETS)).astype(int)
    indices = np.clip(indices, 0, np.array(BUCKETS) - 1)
    return tuple(indices)


def train():
    env = gym.make("CartPole-v1")

    # Q-Table: shape = (*BUCKETS, n_actions)
    q_table = np.zeros(BUCKETS + (env.action_space.n,))

    epsilon   = EPSILON_START
    rewards   = []

    for ep in range(EPISODES):
        obs, _ = env.reset()
        state  = discretize(obs)
        total_reward = 0

        while True:
            # ε-greedy 动作选择
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs)
            done = terminated or truncated

            # Q-Learning 更新
            best_next = np.max(q_table[next_state])
            q_table[state][action] += ALPHA * (
                reward + GAMMA * best_next - q_table[state][action]
            )

            state        = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            avg = np.mean(rewards[-500:])
            print(f"Episode {ep+1:5d} | avg reward (last 500): {avg:6.1f} | ε={epsilon:.3f}")

    env.close()
    return q_table, rewards


def evaluate(q_table: np.ndarray, episodes: int = 10, render: bool = True):
    """用训练好的 Q-Table 跑评估（纯贪心）。"""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    total = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        while True:
            state  = discretize(obs)
            action = int(np.argmax(q_table[state]))
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
    print("=== 训练 CartPole Q-Learning ===")
    q_table, rewards = train()

    print("\n=== 评估 ===")
    evaluate(q_table)
