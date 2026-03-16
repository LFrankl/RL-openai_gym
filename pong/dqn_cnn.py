"""
Pong-v5 — CNN DQN（从原始像素学习打乒乓球）
============================================

【为什么 Pong 是里程碑？】
  Mnih et al. (2013/2015) 在 Nature 发表的论文证明了同一个算法
  可以直接从游戏像素学会玩 49 款 Atari 游戏，无需任何人工特征工程。
  这是深度强化学习的奠基工作，DQN 这个名字也来自于此。

【与之前游戏的本质区别】
  CartPole / MountainCar / LunarLander：
    输入是物理量（位置、速度、角度），低维连续向量，全连接网络即可

  Pong：
    输入是原始像素帧（210×160×3），高维图像，需要 CNN 提取视觉特征

  这一步跨越代表了 RL 从"知道物理量"到"只看屏幕"的能力飞跃。

【观测空间与动作空间】
  原始观测：(210, 160, 3) RGB 图像，uint8
  处理后观测：(4, 84, 84) 灰度帧堆叠，float32

  动作（6个）：
    0 = NOOP      （无动作）
    1 = FIRE      （发球，游戏开始时需要）
    2 = RIGHT     （球拍向右/上）
    3 = LEFT      （球拍向左/下）
    4 = RIGHTFIRE
    5 = LEFTFIRE

  奖励：
    己方得分 +1，对方得分 -1，其余步骤 0
    → 比 LunarLander 更稀疏，但比 MountainCar 好得多

【图像预处理流程（关键！）】
  原始帧 (210,160,3)
      ↓ 灰度化          → (210,160)      去掉颜色，减少计算量
      ↓ 裁剪比分区域    → (170,160)      去掉上方比分栏（无用信息）
      ↓ 缩放            → (84,84)        标准化尺寸（原论文设定）
      ↓ 归一化          → [0.0, 1.0]     数值稳定，加速收敛
      ↓ 帧堆叠（×4）   → (4,84,84)      提供运动信息（单帧看不出球的速度方向）

  帧堆叠（Frame Stacking）是核心技巧：
  单帧图像是静止的，无法判断球向哪个方向运动、速度多快。
  堆叠连续 4 帧相当于给网络提供了"时间差"，让它能感知运动。
  这是将 Pong 转化为 Markov 过程的关键——满足 MDP 的 Markov 性质。

【CNN 网络结构（遵循原论文）】
  输入 (4, 84, 84)
      → Conv(32, 8×8, stride=4)  → ReLU    提取低级特征（边缘、纹理）
      → Conv(64, 4×4, stride=2)  → ReLU    提取中级特征（形状）
      → Conv(64, 3×3, stride=1)  → ReLU    提取高级特征（物体）
      → Flatten
      → Linear(512)              → ReLU
      → Linear(6)                          输出每个动作的 Q 值

【训练注意事项】
  Atari 游戏训练通常需要数百万步（原论文 50M 帧），CPU 上很慢。
  本实现做了以下简化以在 CPU/MPS 上可运行：
  - 回放池缩小到 50K（原论文 1M）
  - 训练步数缩小到 500K 步（原论文 50M）
  - 仍可学到有效策略，但水平低于原论文

  在 Mac M4 上将 DEVICE 改为 "mps" 可显著加速。
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py

# ALE 环境需要手动注册
gym.register_envs(ale_py)


# ─────────────────────────────────────────────
# 超参数
# ─────────────────────────────────────────────
TOTAL_STEPS     = 500_000   # 总训练步数（原论文 50M，此处缩小方便 CPU 运行）
BATCH_SIZE      = 32        # 原论文使用 32
GAMMA           = 0.99
LR              = 1e-4      # Atari 任务学习率比 LunarLander 更小，图像梯度噪声大
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 500_000   # ε 线性衰减的步数跨度（非指数，原论文用线性）
REPLAY_SIZE     = 50_000    # 原论文 1M，缩小以适应内存
TARGET_UPDATE   = 1_000     # 每隔多少步同步目标网络（原论文 10000）
MIN_REPLAY      = 10_000    # 开始训练前最少积累的经验数
FRAME_SKIP      = 4         # 每个动作重复执行帧数（ALE 环境内置）
STACK_FRAMES    = 4         # 堆叠帧数，提供运动信息

DEVICE = torch.device("cpu")
# DEVICE = torch.device("mps")  # Mac M4 启用 GPU 加速（取消注释即可）


# ─────────────────────────────────────────────
# 图像预处理
# ─────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    将原始 RGB 帧 (210,160,3) 处理为灰度单帧 (84,84)。

    步骤：
      1. 灰度化：加权平均 R/G/B 通道（人眼对绿色最敏感）
         gray = 0.299R + 0.587G + 0.114B
      2. 裁剪：去掉上方 26 行（比分栏）和下方 14 行（空白）
         保留 [26:196] → (170, 160)
      3. 缩放：双线性插值到 (84, 84)
      4. 归一化到 [0.0, 1.0]
    """
    # 灰度化（手动加权，避免引入 cv2 依赖）
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

    # 裁剪掉比分区域
    gray = gray[26:196, :]   # (170, 160)

    # 缩放到 84×84（简单的步长采样，速度快）
    # 等价于 resize(84,84)，对 Atari 精度足够
    gray = gray[::2, ::2]    # (85, 80) 近似，再裁到 84×84
    gray = gray[:84, :80]    # 先裁高度

    # 补宽度：80 → 84，用 np.pad 在两侧各填 2 列
    gray = np.pad(gray, ((0, 0), (2, 2)), mode='edge')  # (84, 84)

    # 归一化
    return gray / 255.0


class FrameStack:
    """
    帧堆叠器：维护最近 n 帧，拼接为网络输入。

    内部用 deque 滑动窗口存储最近 n 帧，
    每步只更新最新一帧，其余帧保持不变。

    输出 shape：(n, 84, 84)，对应 CNN 的输入通道数。
    """
    def __init__(self, n: int = 4):
        self.n  = n
        self.frames = deque(maxlen=n)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """episode 开始时用同一帧填满所有槽位（没有历史帧可用）。"""
        for _ in range(self.n):
            self.frames.append(frame)
        return self._get()

    def step(self, frame: np.ndarray) -> np.ndarray:
        """每步推入新帧，自动丢弃最旧的。"""
        self.frames.append(frame)
        return self._get()

    def _get(self) -> np.ndarray:
        """将 deque 中的帧堆叠为 (n, 84, 84) 数组。"""
        return np.stack(self.frames, axis=0)


# ─────────────────────────────────────────────
# CNN Q 网络（遵循 Mnih et al. 2015 原论文结构）
# ─────────────────────────────────────────────
class CNNQNetwork(nn.Module):
    """
    输入：(batch, 4, 84, 84) — 4 帧灰度图堆叠
    输出：(batch, 6)         — 每个动作的 Q 值

    三层卷积负责从像素中提取视觉特征，
    全连接层将特征映射为 Q 值。

    参数量约 1.7M，比 LunarLander 的全连接网络（~132K）大 10 倍以上。
    这也是为什么 Atari 训练需要更多步数和更大内存。
    """
    def __init__(self, n_frames: int, n_actions: int):
        super().__init__()

        # 卷积部分：提取空间特征
        # 每层 kernel size 和 stride 参照原论文
        self.conv = nn.Sequential(
            # 第一层：大 kernel（8×8）+ 大步长（4）快速降维
            # 捕捉大范围低级特征（边缘、球的位置）
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),  # → (32, 20, 20)
            nn.ReLU(),

            # 第二层：中等 kernel（4×4）继续提取
            nn.Conv2d(32, 64, kernel_size=4, stride=2),         # → (64, 9, 9)
            nn.ReLU(),

            # 第三层：小 kernel（3×3）精细特征
            nn.Conv2d(64, 64, kernel_size=3, stride=1),         # → (64, 7, 7)
            nn.ReLU(),
        )

        # 计算卷积输出的展平维度
        conv_out = 64 * 7 * 7  # = 3136

        # 全连接部分：将视觉特征映射为 Q 值
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 4, 84, 84)，值域 [0, 1]
        返回: (batch, n_actions) Q 值
        """
        # 卷积提取特征
        feat = self.conv(x)

        # 展平为一维向量
        feat = feat.view(feat.size(0), -1)  # (batch, 3136)

        # 全连接输出 Q 值
        return self.fc(feat)


# ─────────────────────────────────────────────
# 经验回放缓冲区
# ─────────────────────────────────────────────
class ReplayBuffer:
    """
    与 LunarLander 版本相比，Pong 的回放缓冲区存储图像数据，
    内存占用更大：50K 条 × (4×84×84) float32 ≈ 1.1 GB。
    实际使用 uint8 存储可压缩到 ~280 MB，此处为简化直接用 float32。
    """
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32, device=DEVICE),
            torch.tensor(actions,            dtype=torch.long,    device=DEVICE),
            torch.tensor(rewards,            dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=DEVICE),
            torch.tensor(dones,              dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────
# ε 线性衰减（Pong 用线性，不用指数）
# ─────────────────────────────────────────────
def get_epsilon(step: int) -> float:
    """
    原论文使用线性衰减：
      前 EPSILON_DECAY 步从 1.0 线性降到 0.05，之后保持 0.05。

    区别于之前游戏的指数衰减（每回合乘以系数）：
    线性衰减以"步数"为单位，对长 episode 更公平。
    """
    if step >= EPSILON_DECAY:
        return EPSILON_END
    return EPSILON_START - (EPSILON_START - EPSILON_END) * step / EPSILON_DECAY


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def train():
    env        = gym.make("ALE/Pong-v5", frameskip=FRAME_SKIP)
    n_actions  = env.action_space.n  # 6
    stacker    = FrameStack(STACK_FRAMES)

    policy_net = CNNQNetwork(STACK_FRAMES, n_actions).to(DEVICE)
    target_net = CNNQNetwork(STACK_FRAMES, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
    replay     = ReplayBuffer(REPLAY_SIZE)

    total_steps  = 0
    ep           = 0
    ep_rewards   = []

    print(f"网络参数量: {sum(p.numel() for p in policy_net.parameters()):,}")
    print(f"开始训练，目标 {TOTAL_STEPS:,} 步...\n")

    while total_steps < TOTAL_STEPS:
        raw_obs, _  = env.reset()
        # 预处理第一帧，用它初始化帧堆叠器（填满 4 个槽）
        obs         = stacker.reset(preprocess(raw_obs))
        ep_reward   = 0
        epsilon     = get_epsilon(total_steps)

        while True:
            # ── 动作选择 ─────────────────────────────
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    t      = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = int(policy_net(t).argmax(dim=1).item())

            raw_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 预处理新帧并推入帧堆叠器
            next_obs = stacker.step(preprocess(raw_next))

            # Pong 奖励裁剪：原论文将奖励裁剪到 {-1, 0, +1}
            # 理由：统一不同游戏的奖励量级，让同一组超参数适用于所有游戏
            clipped_reward = np.sign(reward)

            replay.push(obs, action, clipped_reward, next_obs, float(done))
            obs          = next_obs
            ep_reward   += reward  # 记录原始奖励（用于评估）
            total_steps += 1

            # ── 网络更新 ─────────────────────────────
            if len(replay) >= MIN_REPLAY:
                s, a, r, s_, d = replay.sample(BATCH_SIZE)

                # Double DQN 的 TD 目标（减少 Q 值高估）
                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_actions = policy_net(s_).argmax(dim=1, keepdim=True)
                    next_q       = target_net(s_).gather(1, next_actions).squeeze(1)
                    td_target    = r + GAMMA * next_q * (1.0 - d)

                loss = nn.functional.mse_loss(q_values, td_target)
                optimizer.zero_grad()
                loss.backward()
                # Atari 原论文用梯度裁剪，防止 CNN 梯度爆炸
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

            # 定期同步目标网络（按步数，不按回合）
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        ep += 1
        ep_rewards.append(ep_reward)

        if ep % 20 == 0:
            avg     = np.mean(ep_rewards[-20:])
            epsilon = get_epsilon(total_steps)
            print(f"Episode {ep:4d} | steps: {total_steps:7,} | avg(20): {avg:6.1f} | ε={epsilon:.3f}")

    env.close()
    return policy_net


# ─────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────
def evaluate(policy_net: CNNQNetwork, episodes: int = 5, render: bool = True):
    """
    纯贪心策略评估。
    Pong 得分范围：-21（全败）到 +21（全胜）。
    达到正分（>0）即可认为开始击败对手。
    """
    env     = gym.make("ALE/Pong-v5", frameskip=FRAME_SKIP,
                       render_mode="human" if render else None)
    stacker = FrameStack(STACK_FRAMES)
    total   = 0

    policy_net.eval()
    with torch.no_grad():
        for ep in range(episodes):
            raw_obs, _ = env.reset()
            obs        = stacker.reset(preprocess(raw_obs))
            ep_reward  = 0

            while True:
                t      = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = int(policy_net(t).argmax(dim=1).item())
                raw_next, reward, terminated, truncated, _ = env.step(action)
                obs        = stacker.step(preprocess(raw_next))
                ep_reward += reward
                if terminated or truncated:
                    break

            total += ep_reward
            print(f"  评估回合 {ep+1}: {ep_reward:+.0f}")

    env.close()
    avg = total / episodes
    print(f"\n平均得分: {avg:+.1f}  (正分=开始击败对手，+21=完胜)")
    return avg


if __name__ == "__main__":
    print("=" * 50)
    print("Pong CNN DQN")
    print("=" * 50)
    print("【提示】Atari 训练在 CPU 上较慢，建议：")
    print("  - Mac M4：将 DEVICE 改为 'mps' 加速约 5-10x")
    print("  - 先跑完训练再看评估动画\n")

    policy_net = train()

    print("\n=== 评估（开启动画）===")
    evaluate(policy_net)
