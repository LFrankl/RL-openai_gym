# BipedalWalker 算法理论分析

## 问题形式化

BipedalWalker-v3 是一个连续状态、**连续动作**的马尔可夫决策过程：

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

- **状态空间** $\mathcal{S} \subseteq \mathbb{R}^{24}$：关节角度、角速度、激光雷达测距（10条）、地面接触等
- **动作空间** $\mathcal{A} = [-1, 1]^4$：4 个关节的连续力矩（髋关节 ×2、膝关节 ×2）
- **奖励函数**：$R = r_{\text{前进}} - 0.003\|a\|^2 - \mathbf{1}_{\text{摔倒}} \cdot 100$
- **终止条件**：机身高度过低（摔倒），或走完赛道
- **解决标准**：连续 100 回合均分 $\geq 300$

### 与前三个游戏的根本差异

| 游戏 | 动作空间 | 关键挑战 |
|------|---------|---------|
| CartPole | $\{0,1\}$（离散） | 稠密奖励，入门 |
| MountainCar | $\{0,1,2\}$（离散） | 稀疏奖励，探索 |
| LunarLander | $\{0,1,2,3\}$（离散） | 多目标，值函数高估 |
| **BipedalWalker** | $[-1,1]^4$（**连续**） | **argmax 无解，需要连续控制算法** |

连续动作空间使 DQN 的核心操作 $\arg\max_{a} Q(s,a)$ 失效——连续空间无法穷举，也没有解析的最大值。

---

## 一、确定性策略梯度（DPG）——TD3 的理论基础

### 1.1 从随机策略梯度到确定性策略梯度

随机策略梯度定理（Sutton et al., 1999）：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim \rho^\pi,\, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a) \right]$$

对连续动作空间，Silver et al. (2014) 证明确定性策略 $\mu_\theta: \mathcal{S} \to \mathcal{A}$ 存在对应的**确定性策略梯度（DPG）定理**：

$$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^\mu} \left[ \nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^\mu(s,a)\Big|_{a=\mu_\theta(s)} \right]$$

**关键区别**：
- 随机策略梯度需要对动作分布积分（高方差）
- 确定性策略梯度不需要对动作积分，只在策略输出的单点 $a=\mu_\theta(s)$ 处计算 Q 值梯度（低方差）

链式法则展开：

$$\nabla_\theta J = \mathbb{E}_s \left[ \nabla_a Q(s,a)\Big|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s) \right]$$

梯度先沿 $Q$ 对 $a$ 的方向（Q 值增长最快的动作方向），再通过策略网络反传到参数 $\theta$。

### 1.2 DDPG 与过估计问题

深度确定性策略梯度（DDPG，Lillicrap et al., 2015）将 DPG 与 DQN 的经验回放和目标网络结合。但存在严重的 **Q 值过估计**问题：

确定性策略下，TD 目标为：

$$y = r + \gamma Q_{\bar\theta}(s', \mu_{\bar\phi}(s'))$$

策略总是选择估计值最高的动作，系统性地利用 $Q$ 函数的正误差，导致 $Q$ 值无界上升。

### 1.3 TD3 的三个修正

**修正 1：双 Critic**

训练两个独立 Q 网络 $Q_{\theta_1}, Q_{\theta_2}$，TD 目标取 min：

$$y = r + \gamma (1-d) \cdot \min_{i=1,2} Q_{\bar\theta_i}(s', \tilde{a}')$$

设 $Q_1 = Q^* + \epsilon_1$，$Q_2 = Q^* + \epsilon_2$，$\epsilon_i \sim \text{i.i.d.}$，则：

$$\mathbb{E}[\min(Q_1, Q_2)] \leq Q^* \leq \mathbb{E}[\max(Q_1, Q_2)]$$

min 操作将过估计转为欠估计（underestimation），欠估计对策略学习危害小得多。

**修正 2：目标策略平滑**

TD 目标中的 $Q'(s', \mu'(s'))$ 对 $\mu'$ 在尖峰附近非常敏感。TD3 在目标动作上加截断高斯噪声：

$$\tilde{a}' = \text{clip}\!\left(\mu_{\bar\phi}(s') + \text{clip}(\varepsilon, -c, c),\; -1, 1\right), \quad \varepsilon \sim \mathcal{N}(0, \sigma)$$

等价于用 $Q'$ 在目标动作邻域上的期望代替点估计：

$$\mathbb{E}_\varepsilon[Q'(s', \tilde{a}')] \approx \int Q'(s', a') \cdot \mathcal{N}(a'; \mu'(s'), \sigma^2)\, da'$$

平滑后 $Q'$ 在动作空间上的锐峰被抹平，减少了 TD 目标的方差。

**修正 3：延迟策略更新**

Critic 每步更新，Actor 每 $d$ 步更新一次（默认 $d=2$）。

直觉：若 Critic 还未收敛，它提供的梯度方向 $\nabla_a Q$ 是错误的，过早更新 Actor 会让策略朝错误方向走。延迟更新让 Critic 先稳定，再指导 Actor。

---

## 二、最大熵强化学习——SAC 的理论基础

### 2.1 最大熵目标

标准 RL 只最大化期望回报。SAC（Haarnoja et al., 2018）在每步额外奖励策略的**香农熵**：

$$J_{\text{SAC}}(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t \Big( R(s_t, a_t) + \alpha \cdot H(\pi(\cdot|s_t)) \Big)\right]$$

其中 $H(\pi(\cdot|s)) = -\mathbb{E}_{a\sim\pi}[\log \pi(a|s)]$ 是策略的香农熵，$\alpha > 0$ 是温度参数。

**最大熵框架的理论优势**：

设 $\Pi$ 为策略集合，最优最大熵策略满足：

$$\pi^*_{\text{MaxEnt}} = \arg\max_{\pi \in \Pi} J_{\text{SAC}}(\pi)$$

当多种策略都能获得相近回报时，最大熵框架偏好"更随机"的策略，这带来更好的探索性和鲁棒性。

### 2.2 软 Bellman 方程

在最大熵目标下，Bellman 方程变为**软 Bellman 方程**：

$$Q_{\text{soft}}^*(s,a) = R(s,a) + \gamma\, \mathbb{E}_{s'}\!\left[\mathbb{E}_{a'\sim\pi^*}\!\left[Q_{\text{soft}}^*(s',a') - \alpha \log \pi^*(a'|s')\right]\right]$$

最优策略与 Q 函数的关系：

$$\pi^*(a|s) = \exp\!\left(\frac{Q_{\text{soft}}^*(s,a) - V_{\text{soft}}^*(s)}{\alpha}\right)$$

即最优策略是关于 Q 值的 **Boltzmann 分布**（吉布斯分布），温度 $\alpha$ 控制分布的锐度：$\alpha \to 0$ 时退化为确定性贪心策略，$\alpha \to \infty$ 时退化为均匀随机策略。

### 2.3 实际策略：压缩高斯（Squashed Gaussian）

理论上最优策略是 Boltzmann 分布，实际中用参数化的**压缩高斯分布**近似：

$$u \sim \mathcal{N}(\mu_\theta(s),\, \sigma_\theta^2(s)), \quad a = \tanh(u)$$

$\tanh$ 将无界的高斯分布压缩到 $(-1,1)^{act\_dim}$，与动作空间匹配。

对数概率（Jacobian 修正项）：

$$\log \pi(a|s) = \log \mathcal{N}(u|\mu_\theta(s), \sigma_\theta(s)) - \sum_i \log(1 - \tanh^2(u_i))$$

减去的 $\log(1-\tanh^2(u))$ 是 $\tanh$ 变换的对数 Jacobian 行列式，保证概率密度在变量替换下正确归一化。

### 2.4 重参数化技巧（Reparameterization Trick）

对 Actor 的损失 $\mathcal{L}(\phi) = -\mathbb{E}_{a\sim\pi_\phi}[Q(s,a) - \alpha \log \pi_\phi(a|s)]$，
直接对采样动作求梯度会引入高方差（REINFORCE 方差问题）。

重参数化将随机性转移到与参数无关的噪声 $\varepsilon$：

$$a = f_\phi(s, \varepsilon) = \tanh(\mu_\phi(s) + \sigma_\phi(s) \cdot \varepsilon), \quad \varepsilon \sim \mathcal{N}(0, I)$$

此时梯度可以直接通过 $f_\phi$ 反传：

$$\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{\varepsilon}\!\left[Q(s, f_\phi(s,\varepsilon)) - \alpha \log \pi_\phi(f_\phi(s,\varepsilon)|s)\right]$$

期望内的函数对 $\phi$ 可微，梯度方差与蒙特卡洛估计的方差同阶，远低于 REINFORCE。

### 2.5 自动温度调节

手动调节 $\alpha$ 依赖经验。SAC 将其转化为带约束的优化问题：

$$\max_{\pi} J_{\text{SAC}}(\pi) \quad \text{s.t.} \quad \mathbb{E}_{a\sim\pi(\cdot|s)}[-\log\pi(a|s)] \geq \mathcal{H}_{\text{target}}$$

即策略熵不得低于目标熵 $\mathcal{H}_{\text{target}}$。用 Lagrange 乘数法，乘数 $\alpha \geq 0$ 即为温度，对应的 $\alpha$ 更新规则为：

$$\mathcal{L}(\alpha) = \mathbb{E}_{a\sim\pi}\!\left[-\alpha\log\pi(a|s) - \alpha\mathcal{H}_{\text{target}}\right]$$

梯度方向：若策略熵 $< \mathcal{H}_{\text{target}}$，则 $\nabla_\alpha \mathcal{L} > 0$，$\alpha$ 增大，鼓励更多探索；反之减小。

理论推荐 $\mathcal{H}_{\text{target}} = -|\mathcal{A}|$（动作维度的负数）。对 BipedalWalker，$\mathcal{H}_{\text{target}} = -4$。

---

## 三、SAC vs TD3 深度对比

### 3.1 策略类型

| 维度 | TD3（确定性） | SAC（随机） |
|------|-------------|------------|
| 策略表示 | $a = \mu_\phi(s)$ | $a \sim \pi_\phi(\cdot\|s) = \tanh(\mathcal{N}(\mu,\sigma))$ |
| 探索方式 | 外加噪声 $a+\varepsilon$，需要手动调 | 策略自带随机性，自动调节（通过 $\alpha$） |
| 梯度计算 | DPG，直接链式法则 | 重参数化，低方差但多一步采样 |
| 评估时策略 | 直接 $\mu(s)$，无噪声 | $\tanh(\mu(s))$，用均值（确定性化） |

### 3.2 Q 值高估处理

| 方法 | TD3 | SAC |
|------|-----|-----|
| 双 Critic | ✓ min(Q1, Q2) | ✓ min(Q1, Q2) |
| TD 目标附加项 | 目标策略平滑（加噪声） | 熵项 $-\alpha\log\pi$（自然正则化） |

SAC 的熵项 $-\alpha\log\pi$ 对 Q 值做了隐式的正则化——策略不会完全确定性，Q 值永远反映的是分布下的期望，而非单点最大值，天然减少高估。

### 3.3 超参数敏感性

TD3 的关键超参数：
- $\sigma_{\text{explore}}$（探索噪声）：太小不探索，太大动作不稳定
- $\sigma_{\text{target}}$（目标平滑噪声）：影响 Q 值平滑程度
- $d$（策略延迟）：一般设 2，对结果不太敏感

SAC 的关键超参数：
- $\alpha$ 或 $\mathcal{H}_{\text{target}}$：自动调节后几乎不需要人工干预

SAC 通过自动温度调节大幅降低了超参数调优难度，这是其在实践中更受欢迎的主要原因之一。

---

## 四、实验对比

（待填入实际训练结果）

| 算法 | 评估均分 | 收敛回合 | 备注 |
|------|---------|---------|------|
| TD3 | — | — | 确定性策略，探索噪声需调 |
| SAC | — | — | 随机策略，自动温度调节 |

> 详细训练曲线待补充。

---

## 五、为什么需要这两个算法——与前序工作的联系

```
离散动作空间              连续动作空间
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q-Learning / DQN          →  DDPG（确定性策略梯度）
  argmax 可穷举                  DPG 定理，链式法则
     ↓                                ↓
Double DQN                  TD3（解决过估计、不稳定）
  消除选择-评估耦合            双Critic + 目标平滑 + 延迟更新
     ↓                                ↓
Dueling DQN                SAC（最大熵，自动探索）
  V+A 分解，更高效学习           软Bellman方程 + 重参数化 + 自动α
```

BipedalWalker 是这条演进链上的重要一环：从"能枚举动作"到"动作空间无法枚举"，
算法设计的核心矛盾从探索（MountainCar）、高估（LunarLander）转向了**如何在连续空间中有效更新策略**。

---

## 参考文献

- Silver, D., et al. (2014). *Deterministic Policy Gradient Algorithms*. ICML.
- Lillicrap, T., et al. (2015). *Continuous Control with Deep Reinforcement Learning (DDPG)*. ICLR 2016.
- Fujimoto, S., et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods (TD3)*. ICML.
- Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning*. ICML.
- Haarnoja, T., et al. (2018). *Soft Actor-Critic Algorithms and Applications*. arXiv:1812.05905.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
