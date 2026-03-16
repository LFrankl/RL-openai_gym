# rl-gym

用 [Gymnasium](https://gymnasium.farama.org/) 学强化学习，从经典控制问题入手。

## 目录结构

```
rl-gym/
├── cartpole/
│   ├── q_learning.py   # 表格法 Q-Learning，离散化状态空间
│   └── dqn.py          # Deep Q-Network，神经网络直接处理连续状态
├── pyproject.toml
├── uv.lock
└── .python-version
```

## 示例：CartPole-v1

小车上竖着一根杆，每步决定往左还是往右推，目标是让杆不倒。每撑过一步得 1 分，最高 500 分。

| 算法 | 原理 | 600回合均分 |
|------|------|------------|
| Q-Learning | 状态离散化 + Q-Table | ~155 |
| DQN | 神经网络拟合 Q 值 + 经验回放 | ~277 |

## 快速上手

参考 [DEPLOY.md](./DEPLOY.md)。
