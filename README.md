# rl-gym

用 [Gymnasium](https://gymnasium.farama.org/) 学强化学习，从经典控制问题入手。

## 目录结构

```
rl-gym/
├── cartpole/
│   ├── q_learning.py       # 表格法 Q-Learning，离散化状态空间
│   ├── dqn.py              # Deep Q-Network，神经网络直接处理连续状态
│   ├── ppo.py              # PPO，Actor-Critic + GAE + Clip
│   └── THEORY.md           # 三种算法的严格理论分析
├── mountaincar/
│   ├── q_learning_naive.py # 原始 Q-Learning，展示稀疏奖励下的失败
│   ├── q_learning_shaped.py# 奖励塑形 + 乐观初始化，仍失败
│   ├── ppo_shaped.py       # PPO + 多种奖励设计，仍失败
│   ├── dqn_shaped.py       # 行为克隆预热，100% 到顶
│   └── THEORY.md           # 8 次尝试的完整失败分析 + BC 理论
├── pyproject.toml
├── uv.lock
└── .python-version
```

## CartPole-v1

小车上竖着一根杆，每步决定往左还是往右推，目标是让杆不倒。每撑过一步得 +1 分，最高 500 分。**奖励稠密**，适合入门。

| 算法 | 原理 | 评估均分 |
|------|------|---------|
| Q-Learning | 状态离散化 + Q-Table | ~351 |
| DQN | 神经网络 + 经验回放 + Target Net | ~190 |
| PPO | Actor-Critic + GAE + Clip | ~212 |

## MountainCar-v0

小车在山谷中，动力不足以直接冲顶，必须左右摇摆积累势能。每步 -1 分，到顶终止。**奖励极稀疏**，是 RL 经典探索难题。

随机策略在 10,000 回合内到达山顶的次数：**0 次**。

| 算法 | 方法 | 到顶率 | 均分 |
|------|------|--------|------|
| Q-Learning | ε-greedy | 失败 | -200 |
| Q-Learning | 奖励塑形 + 乐观初始化 | 失败 | -200 |
| DQN | 差分势函数 / 自定义奖励 / Bootstrap | 失败 | -200 |
| PPO | 速度势 / 高度势 / 自定义奖励 | 失败 | -200 |
| **行为克隆（BC）** | **专家演示 + 监督学习** | **100%** | **-118** |

> 详细的失败原因分析见 [mountaincar/THEORY.md](./mountaincar/THEORY.md)。

## 快速上手

参考 [DEPLOY.md](./DEPLOY.md)。
