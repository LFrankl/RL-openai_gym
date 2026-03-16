# CartPole 算法理论分析

## 问题形式化

CartPole-v1 是一个有限水平马尔可夫决策过程（MDP），形式化为五元组：

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

- **状态空间** $\mathcal{S} \subseteq \mathbb{R}^4$：小车位置 $x$、小车速度 $\dot{x}$、杆角度 $\theta$、杆角速度 $\dot{\theta}$
- **动作空间** $\mathcal{A} = \{0, 1\}$：离散二值（左/右施力）
- **转移函数** $P(s' \mid s, a)$：由牛顿力学确定，实际为确定性系统
- **奖励函数** $R(s, a) = 1$（每步存活得 1 分，终止得 0 分）
- **折扣因子** $\gamma \in [0, 1)$

**目标**：找到策略 $\pi^* : \mathcal{S} \to \mathcal{A}$，最大化期望累计回报：

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]$$

---

## 一、Q-Learning（表格法）

### 1.1 Bellman 最优方程

定义动作价值函数：

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \,\Big|\, s_0=s,\, a_0=a \right]$$

最优动作价值函数 $Q^*$ 满足 **Bellman 最优方程**：

$$Q^*(s, a) = \mathbb{E}_{s'} \left[ R(s,a) + \gamma \max_{a'} Q^*(s', a') \right]$$

这是一个不动点方程。定义 Bellman 最优算子 $\mathcal{T}$：

$$(\mathcal{T}Q)(s,a) = \mathbb{E}_{s'}\left[ R(s,a) + \gamma \max_{a'} Q(s',a') \right]$$

可以证明 $\mathcal{T}$ 是关于 $\sup$-范数的 $\gamma$-收缩映射，即：

$$\|\mathcal{T}Q_1 - \mathcal{T}Q_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$$

由 Banach 不动点定理，$\mathcal{T}$ 存在唯一不动点 $Q^*$。

### 1.2 Q-Learning 更新规则

Q-Learning 以随机近似的方式逼近该不动点：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ \underbrace{r_t + \gamma \max_{a'} Q(s_{t+1}, a')}_{\text{TD 目标}} - Q(s_t, a_t) \right]$$

方括号内称为 **TD 误差**（Temporal Difference Error）：

$$\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$$

### 1.3 收敛性

**定理**（Watkins & Dayan, 1992）：若满足以下条件，Q-Learning 以概率 1 收敛到 $Q^*$：

1. 所有状态-动作对被无限次访问：$\sum_t \mathbf{1}[(s_t, a_t) = (s, a)] = \infty$
2. 学习率满足 Robbins-Monro 条件：$\sum_t \alpha_t = \infty$，$\sum_t \alpha_t^2 < \infty$
3. 奖励有界：$|R| < \infty$

### 1.4 状态离散化的代价

连续状态空间需要手工分桶。设每维分 $b$ 个桶，$d$ 维状态，Q-Table 大小为 $b^d \cdot |\mathcal{A}|$。

**量化误差**：设真实 Q 函数 Lipschitz 连续（常数 $L$），桶宽为 $\epsilon$，则离散化引入的近似误差上界为：

$$\|Q^* - \hat{Q}^*\|_\infty \leq \frac{L \epsilon}{1 - \gamma}$$

其中 $\hat{Q}^*$ 为离散化后的最优 Q 值。桶越细误差越小，但表格指数增长——即**维数灾难**。

---

## 二、DQN（Deep Q-Network）

### 2.1 函数逼近

用参数为 $\theta$ 的神经网络 $Q_\theta(s, a)$ 代替 Q-Table，将 Bellman 最优方程转化为回归问题，最小化：

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

其中 $\mathcal{D}$ 为经验回放缓冲区，$\bar{\theta}$ 为目标网络参数（固定）。

### 2.2 经验回放（Experience Replay）

**问题根源**：在线学习时，连续采集的样本 $(s_t, a_t, r_t, s_{t+1})$ 具有强时间相关性，违反随机梯度下降对 i.i.d. 样本的假设，导致训练不稳定。

**解决方案**：维护缓冲区 $\mathcal{D}$，存储历史转移。每次更新从 $\mathcal{D}$ 中均匀随机采样 mini-batch：

$$\nabla_\theta \mathcal{L}(\theta) \approx \frac{1}{|B|} \sum_{(s,a,r,s') \in B} \delta \cdot \nabla_\theta Q_\theta(s, a)$$

采样后样本近似独立，满足 SGD 假设。此外，每条经验可被多次复用，提升样本效率。

### 2.3 目标网络（Target Network）

**问题根源**：若用同一网络计算 TD 目标和当前 Q 值，目标随参数变化而移动，形成"移动靶"，梯度更新方向不稳定：

$$\nabla_\theta \mathcal{L} = -\mathbb{E}\left[\delta_t \cdot \nabla_\theta Q_\theta(s_t, a_t)\right], \quad \delta_t \text{ 也依赖 } \theta$$

**解决方案**：使用独立的目标网络 $Q_{\bar{\theta}}$，每隔 $C$ 步同步一次 $\bar{\theta} \leftarrow \theta$，在两次同步之间 TD 目标固定，梯度方向稳定。

### 2.4 致命三角问题

函数逼近 + 自举（Bootstrapping）+ 离策略学习同时存在时，Q 值可能发散。这被称为**致命三角**（Deadly Triad，Sutton & Barto）。经验回放和目标网络从工程上缓解了这一问题，但不能从理论上保证收敛。

### 2.5 近似误差传播

设函数逼近误差上界为 $\epsilon_F$，则最优策略与 DQN 策略的性能差距满足：

$$\|Q^* - Q_\theta\|_\infty \leq \frac{\epsilon_F}{1 - \gamma} + \frac{\gamma \epsilon_{TD}}{(1-\gamma)^2}$$

其中 $\epsilon_{TD}$ 为 TD 更新的残差。可见折扣因子 $\gamma$ 越大，误差放大效应越强。

---

## 三、PPO（Proximal Policy Optimization）

### 3.1 策略梯度定理

对于参数化策略 $\pi_\theta(a \mid s)$，目标函数为：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$

**策略梯度定理**（Sutton et al., 1999）给出梯度的解析形式：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]$$

实践中用优势函数 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 替换 $Q^\pi$，在不改变梯度期望的前提下降低方差：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right]$$

### 3.2 重要性采样与 TRPO

若用旧策略 $\pi_{\theta_{\text{old}}}$ 收集的数据来更新新策略 $\pi_\theta$，需引入重要性权重：

$$L^{IS}(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} A_t \right] = \mathbb{E}_t \left[ r_t(\theta) \cdot A_t \right]$$

其中 $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ 为**概率比**。

直接最大化 $L^{IS}$ 可能导致策略更新幅度过大，破坏策略单调改进性质。TRPO（Schulman et al., 2015）通过 KL 散度约束解决：

$$\max_\theta \, L^{IS}(\theta) \quad \text{s.t.} \quad \mathbb{E}_t \left[ D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \right] \leq \delta$$

TRPO 需要计算 Fisher 信息矩阵的逆，计算代价高。

### 3.3 PPO Clip 目标

PPO 用一阶方法近似 TRPO 的约束，直接 clip 概率比：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

**几何意义**：
- 当 $A_t > 0$（当前动作优于基线）：$r_t$ 被限制在 $[-, 1+\epsilon]$，防止过度增大概率
- 当 $A_t < 0$（当前动作劣于基线）：$r_t$ 被限制在 $[1-\epsilon, -]$，防止过度减小概率

取 min 保证目标函数是原始目标的下界（pessimistic bound），梯度只在 clip 未激活时才更新策略。

### 3.4 广义优势估计（GAE）

单步 TD 误差（低方差高偏差）与蒙特卡洛回报（低偏差高方差）之间做权衡：

$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE（Schulman et al., 2016）以 $\lambda$ 为权重对不同步长的估计做指数加权：

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

等价于将有效折扣因子从 $\gamma$ 变为 $\gamma\lambda$，偏差-方差权衡由 $\lambda$ 控制：
- $\lambda = 0$：纯单步 TD，高偏差低方差
- $\lambda = 1$：蒙特卡洛，低偏差高方差

### 3.5 完整 PPO 目标

同时优化策略损失、价值函数损失和熵正则项：

$$L(\theta) = \mathbb{E}_t \left[ L^{CLIP}_t(\theta) - c_1 \underbrace{\left( V_\theta(s_t) - V_t^{\text{target}} \right)^2}_{价值函数损失} + c_2 \underbrace{H[\pi_\theta(\cdot \mid s_t)]}_{熵正则} \right]$$

其中 $c_1, c_2$ 为系数，熵项 $H$ 鼓励探索，防止策略过早收敛到次优确定性策略。

### 3.6 单调策略改进保证

**定理**（Schulman et al., 2015）：若每步更新满足 $D_{KL}^{\max}(\pi_{\text{old}} \| \pi_{\text{new}}) \leq \delta$，则策略性能单调不降：

$$J(\pi_{\text{new}}) \geq L_{\pi_{\text{old}}}(\pi_{\text{new}}) - \frac{4\epsilon\gamma}{(1-\gamma)^2} \sqrt{\frac{\delta}{2}}$$

PPO clip 是对该约束的实用化近似，不保证严格单调，但工程上稳定性接近 TRPO。

---

## 四、算法对比

| 维度 | Q-Learning | DQN | PPO |
|------|-----------|-----|-----|
| 策略类型 | 值函数导出（间接） | 值函数导出（间接） | 参数化策略（直接） |
| 状态空间 | 离散（需手工分桶） | 连续 | 连续 |
| 动作空间 | 离散 | 离散 | 离散/连续均可 |
| 样本效率 | 低（在线） | 中（经验回放） | 低（on-policy） |
| 收敛理论 | 有严格保证 | 无（致命三角） | 近似单调改进 |
| 超参数敏感性 | 低 | 中 | 中 |
| 适用规模 | 低维、简单任务 | 中等复杂度 | 中高复杂度、连续控制 |

### 核心区别

**值函数方法 vs 策略梯度方法**：Q-Learning 和 DQN 通过学习 $Q^*$ 间接推导策略（$\pi(s) = \arg\max_a Q(s,a)$），在离散动作空间高效但难以扩展到连续动作。PPO 直接参数化并优化策略，对动作空间类型更灵活，但属于 on-policy 方法，样本利用率低。

**离策略 vs 在策略**：DQN 是 off-policy（可复用历史数据），PPO 是 on-policy（每次更新后数据作废）。前者样本效率高，后者训练更稳定。

---

## 参考文献

- Watkins, C. J., & Dayan, P. (1992). *Q-learning*. Machine Learning.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Schulman, J., et al. (2015). *Trust Region Policy Optimization*. ICML.
- Schulman, J., et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
