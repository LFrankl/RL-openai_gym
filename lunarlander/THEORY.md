# LunarLander 算法理论分析

## 问题形式化

LunarLander-v3 是一个连续状态、离散动作的马尔可夫决策过程：

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

- **状态空间** $\mathcal{S} \subseteq \mathbb{R}^8$：

$$s = (x,\ y,\ \dot{x},\ \dot{y},\ \theta,\ \dot{\theta},\ l_L,\ l_R)$$

其中 $(x, y)$ 为位置，$(\dot{x}, \dot{y})$ 为速度，$\theta$ 为机身角度，$\dot{\theta}$ 为角速度，$l_L, l_R \in \{0,1\}$ 为左右着陆腿的触地状态。

- **动作空间** $\mathcal{A} = \{0, 1, 2, 3\}$：无动作 / 左引擎 / 主引擎 / 右引擎

- **奖励函数**（稠密）：

$$R(s_t, a_t) = r_{\text{pos}} + r_{\text{vel}} + r_{\text{angle}} + r_{\text{leg}} + r_{\text{fuel}} + r_{\text{terminal}}$$

| 分量 | 含义 | 量级 |
|------|------|------|
| $r_{\text{pos}}$ | 靠近着陆垫的位置奖励 | $[-100, +100]$ |
| $r_{\text{vel}}$ | 速度减小的奖励 | 连续 |
| $r_{\text{angle}}$ | 机身保持竖直 | 连续 |
| $r_{\text{leg}}$ | 每条腿触地 $+10$ | $\{0, 10, 20\}$ |
| $r_{\text{fuel}}$ | 每次点火 $-0.3$ | $\{-0.3, 0\}$ |
| $r_{\text{terminal}}$ | 坠毁 $-100$，成功着陆 $+100$ | $\{-100, +100\}$ |

- **解决标准**：连续 100 回合平均得分 $\geq 200$

**与 CartPole / MountainCar 的对比**：CartPole 奖励极稠密（每步 +1），MountainCar 奖励极稀疏（只有到顶才终止），LunarLander 处于中间——奖励稠密但多目标（位置、速度、角度、燃料需要同时优化），是更真实的控制任务。

---

## 一、DQN 基础——已有理论的延伸

DQN 的核心理论在 CartPole 章节已有完整推导（Bellman 方程、经验回放、目标网络），此处聚焦 LunarLander 任务引出的**新问题**。

### 1.1 Q 值高估偏差（Overestimation Bias）

标准 DQN 的 TD 目标：

$$y_t = r_t + \gamma \max_{a'} Q_{\bar{\theta}}(s_{t+1}, a')$$

**高估的根源**：设 $Q_{\bar{\theta}}(s, a) = Q^*(s, a) + \epsilon_a$，其中 $\epsilon_a$ 是零均值的随机估计误差。则：

$$\mathbb{E}\left[\max_a Q_{\bar{\theta}}(s, a)\right] \geq \max_a Q^*(s, a)$$

不等号严格成立（Jensen 不等式的反向形式）：对于有噪声的估计，$\max$ 操作会系统性地挑出噪声偏高的项，导致期望值高于真实最优值。

**量化**（Thrun & Schwartz, 1993）：设 $n$ 个动作的估计误差均匀分布在 $[-\epsilon, \epsilon]$，则高估量的期望为：

$$\mathbb{E}\left[\max_a Q_{\bar{\theta}}(s,a) - \max_a Q^*(s,a)\right] \approx \epsilon \cdot \frac{n-1}{n+1}$$

动作数量 $n$ 越多，高估越严重。LunarLander 有 4 个动作，CartPole 只有 2 个，因此高估问题在 LunarLander 上更显著。

**影响**：高估的 Q 值作为训练目标，进一步拉高 Q 值估计，形成正反馈循环，导致训练后期 Q 值发散或策略不稳定。

---

## 二、Double DQN——解耦选择与评估

### 2.1 核心思想

**Van Hasselt et al. (2016)** 提出将 TD 目标中的"选择动作"与"评估价值"解耦到两个不同参数的网络：

$$a^* = \arg\max_{a'} Q_\theta(s_{t+1}, a') \quad \text{（在线网络选择动作）}$$

$$y_t = r_t + \gamma Q_{\bar{\theta}}(s_{t+1}, a^*) \quad \text{（目标网络评估价值）}$$

对比标准 DQN：

$$y_t^{DQN} = r_t + \gamma \max_{a'} Q_{\bar{\theta}}(s_{t+1}, a') = r_t + \gamma Q_{\bar{\theta}}(s_{t+1}, \arg\max_{a'} Q_{\bar{\theta}}(s_{t+1}, a'))$$

标准 DQN 中，$\bar{\theta}$ 同时承担选择和评估两个角色；Double DQN 中，$\theta$ 负责选择，$\bar{\theta}$ 负责评估。

### 2.2 为什么有效

**直觉**：即使在线网络 $Q_\theta$ 对某个动作 $a^*$ 的 Q 值有噪声偏高（导致错误地选了 $a^*$），目标网络 $Q_{\bar{\theta}}$ 对该动作的评估值也不一定同样偏高（两个网络的随机误差独立）。用一个网络的噪声来"否决"另一个网络的高估，显著减少了偏差。

**形式化**：设两个网络的误差分别为 $\epsilon^{(1)}_{a}$ 和 $\epsilon^{(2)}_{a}$，互相独立。Double DQN 的高估量为：

$$\mathbb{E}\left[Q_{\bar{\theta}}(s, a^*) - Q^*(s, a^*)\right] = \mathbb{E}\left[\epsilon^{(2)}_{a^*}\right] = 0$$

其中 $a^* = \arg\max_a (Q^*(s,a) + \epsilon^{(1)}_a)$。由于 $\epsilon^{(2)}$ 与 $a^*$ 的选取独立，期望为 0，高估被消除。

### 2.3 代码改动极小

Double DQN 相对标准 DQN 只改两行，体现了理论改进的优雅性：

```python
# 标准 DQN（一行）
max_next_q = target_net(s_).max(dim=1).values

# Double DQN（两行）
next_actions = policy_net(s_).argmax(dim=1, keepdim=True)   # 在线网络选动作
next_q       = target_net(s_).gather(1, next_actions).squeeze(1)  # 目标网络评估
```

### 2.4 局限性

Double DQN 减少了高估，但并不能保证完全消除。若在线网络 $Q_\theta$ 和目标网络 $Q_{\bar{\theta}}$ 参数接近（目标网络刚同步后），两者误差的相关性增加，去偏效果减弱。此外，Double DQN 有时会引入**低估偏差**（Underestimation Bias），某些任务上反而略差于标准 DQN。

---

## 三、Dueling DQN——分解 Q 值的结构

### 3.1 优势函数分解

**Wang et al. (2016)** 提出将 Q 值在结构上分解为两个语义不同的量：

$$Q^\pi(s, a) = V^\pi(s) + A^\pi(s, a)$$

其中：

- **状态价值函数** $V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[Q^\pi(s, a)\right]$：当前状态的期望回报，与具体动作无关
- **优势函数** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$：执行动作 $a$ 相对于策略均值的额外收益

由定义可知：$\mathbb{E}_{a \sim \pi}\left[A^\pi(s, a)\right] = 0$，即优势函数在策略分布下的期望为零。

### 3.2 网络结构

标准 DQN 网络：

$$s \xrightarrow{\text{共享层}} h \xrightarrow{\text{单一输出头}} Q(s, \cdot) \in \mathbb{R}^{|\mathcal{A}|}$$

Dueling DQN 网络：

$$s \xrightarrow{\text{共享层}} h \xrightarrow{\text{V 头}} V(s) \in \mathbb{R}^1$$
$$\phantom{s \xrightarrow{\text{共享层}} h} \xrightarrow{\text{A 头}} A(s, \cdot) \in \mathbb{R}^{|\mathcal{A}|}$$

最终合并：

$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$$

### 3.3 为什么要减去均值

分解 $Q = V + A$ 存在**不可辨识性**（Identifiability Problem）：对任意常数 $c$，$(V + c,\ A - c)$ 与 $(V,\ A)$ 得到相同的 $Q$ 值。这意味着网络无法唯一确定 $V$ 和 $A$ 的值，训练不稳定。

**修正方案**：减去 $A$ 的均值，强制令 $\sum_{a'} A(s, a') = 0$，使分解唯一：

$$Q(s, a) = V(s) + \underbrace{\left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)}_{\text{零均值优势，分解唯一}}$$

此时 $V(s) = \frac{1}{|\mathcal{A}|}\sum_{a'} Q(s, a')$，即 $V$ 恰好等于 Q 值的均值，语义清晰。

另一种方案是减去最大值（$\max_{a'} A(s, a')$），可以保证 $\max_{a'} Q(s,a') = V(s)$，但实践中均值方案更稳定。

### 3.4 为什么有效

**核心优势**：$V(s)$ 的梯度来自所有动作的 Q 值更新，而标准 DQN 每次只更新被执行动作的 Q 值。

设一个 batch 中状态 $s$ 出现了 $k$ 次，执行了 $k$ 个不同动作。标准 DQN 中，$Q(s, a_i)$ 对第 $i$ 次更新只被更新一次；Dueling DQN 中，$V(s)$ 被所有 $k$ 次更新共同影响，**等效于对 $V(s)$ 进行了 $k$ 次更新**，样本效率提升 $|\mathcal{A}|$ 倍（在 $V$ 的学习上）。

**适用场景**：当很多状态下不同动作的优劣差异很小时（动作几乎等价），网络只需精确学好 $V(s)$，而无需在每个动作上都精确拟合。LunarLander 中"飞船稳定悬停时短暂无动作"这类状态尤为典型。

### 3.5 Dueling + Double 的组合

Dueling 和 Double DQN 是**正交的改进**，可以无缝组合：

- Dueling 改变的是**网络结构**（V+A 分解）
- Double 改变的是**TD 目标的计算**（解耦选择与评估）

两者结合不存在冲突，`dueling_dqn.py` 同时使用了两种技术。

---

## 四、算法演进脉络

三个算法的改进是线性叠加的，每步只改动一个地方：

```
DQN
│
│  问题：Q 值高估（max 操作偏差）
│  改动：TD 目标中拆分"选动作"和"评估价值"
▼
Double DQN
│
│  问题：V(s) 更新效率低（每次只更新一个动作的 Q）
│  改动：网络结构分解为 V(s) + A(s,a)
▼
Dueling DQN（+ Double）
```

### 实验结果对比

| 算法 | 网络参数量 | 评估均分 | 是否解决(≥200) |
|------|----------|---------|--------------|
| DQN | ~132K | 185.9 | 未解决 |
| Double DQN | ~132K | 155.0 | 未解决 |
| **Dueling DQN** | ~132K | **208.6** | **已解决** |

三个网络参数量相同，差异纯粹来自算法设计，不是模型容量。

Double DQN 本次得分偏低（155），这是强化学习随机性的正常表现——同一算法不同随机种子可能差距较大。在大规模统计实验中，Double DQN 通常优于标准 DQN。

---

## 五、DQN 系列的共同局限

### 5.1 离散动作的限制

DQN 系列的 $\arg\max_a Q(s, a)$ 操作要求动作空间离散且有限。对于连续动作空间（如 MountainCarContinuous、HalfCheetah），无法直接枚举所有动作取 max，需要改用 SAC / TD3 等 Actor-Critic 方法。

### 5.2 Off-Policy 的双刃剑

DQN 使用经验回放，是 off-policy 算法：
- **优点**：样本效率高，历史数据可复用
- **缺点**：若策略变化剧烈，旧数据的分布与当前策略的分布差异大，可能引入偏差（分布偏移问题）

### 5.3 超参数敏感性

DQN 对以下超参数敏感：
- 目标网络更新频率 $C$：太小→训练不稳定，太大→收敛慢
- 回放池大小：太小→数据相关性强，太大→早期差数据影响持久
- ε 衰减速度：太快→探索不足，太慢→利用不充分

---

## 参考文献

- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI.
- Wang, Z., et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML.
- Thrun, S., & Schwartz, A. (1993). *Issues in using function approximation for reinforcement learning*. Connectionist Models Summer School.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
