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
├── lunarlander/
│   ├── dqn.py              # 基础 DQN，引出 Q 值高估问题
│   ├── double_dqn.py       # Double DQN，解耦动作选择与价值评估
│   ├── dueling_dqn.py      # Dueling DQN，V(s)+A(s,a) 网络分解
│   └── THEORY.md           # Q 值高估证明 + 三代算法演进理论
├── pong/
│   ├── dqn_cnn.py          # CNN DQN，完整 Atari 图像预处理管道
│   └── THEORY.md           # 图像 MDP 理论 + CNN 结构分析
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

## LunarLander-v3

飞船从高空下落，目标是稳稳降落在着陆垫上。奖励**稠密但多目标**（位置、速度、角度、燃料同时优化），$\geq 200$ 分视为解决。

三个算法网络参数量相同，差异纯粹来自算法设计：

| 算法 | 改进点 | 评估均分 | 是否解决 |
|------|--------|---------|---------|
| DQN | 基准 | 185.9 | 未解决 |
| Double DQN | 解耦选动作与评估价值，消除 Q 值高估 | 155.0 | 未解决 |
| **Dueling DQN** | **V(s)+A(s,a) 分解，提升 V 的学习效率** | **208.6** | **已解决** |

> 三代算法的演进理论见 [lunarlander/THEORY.md](./lunarlander/THEORY.md)。

## ALE/Pong-v5

Atari Pong，图像像素作为输入。与前三个游戏最大的区别：**状态不再是低维向量，而是 $(210, 160, 3)$ 的 RGB 图像**，需要 CNN 提取特征。

图像输入带来的核心问题：单帧图像不满足 Markov 性质（无法判断球的运动方向）。

| 算法 | 关键技术 | 备注 |
|------|---------|------|
| **CNN DQN** | 帧堆叠 4 帧 + 图像预处理 + Double DQN 目标 | 原论文 5000 万步，本实现 50 万步（基本策略）|

图像预处理管道：灰度化 → 裁剪比分栏 → 缩放 84×84 → 归一化 → 帧堆叠 4 帧

网络结构（Mnih et al., 2015）：$4 \times 84 \times 84 \xrightarrow{\text{Conv}\times 3} \xrightarrow{\text{FC}(512)} Q \in \mathbb{R}^6$，参数量 1.7M

> 算法理论（Markov 性质、CNN 感受野、奖励裁剪等）见 [pong/THEORY.md](./pong/THEORY.md)。

```
pong/
├── dqn_cnn.py   # CNN DQN，完整 Atari 预处理管道
└── THEORY.md    # 图像 MDP 理论 + 预处理必要性分析
```

## BipedalWalker-v3

双足机器人行走，**连续动作空间** $[-1,1]^4$（4 个关节力矩）。与前四个游戏的根本差异在于：DQN 的核心操作 $\arg\max_a Q(s,a)$ 在连续空间无解，必须切换算法范式。

两个算法从不同角度解决连续控制问题：

| 算法 | 策略类型 | 探索方式 | 核心思想 |
|------|---------|---------|---------|
| TD3 | 确定性 $\mu(s)$ | 外加高斯噪声 | 双Critic + 目标策略平滑 + 延迟更新 |
| SAC | 随机 $\pi(a\|s)$ | 策略自带随机性 | 最大熵框架 + 重参数化 + 自动温度调节 |

（评估结果待训练后填入）

> 连续动作理论推导（DPG 定理、最大熵 Bellman 方程、重参数化技巧等）见 [bipedalwalker/THEORY.md](./bipedalwalker/THEORY.md)。

```
bipedalwalker/
├── sac.py         # Soft Actor-Critic，最大熵随机策略
├── td3.py         # Twin Delayed DDPG，确定性策略
└── THEORY.md      # 连续动作空间理论分析
```

## 快速上手

参考 [DEPLOY.md](./DEPLOY.md)。

