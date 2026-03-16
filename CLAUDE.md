# CLAUDE.md — rl-gym 项目规范

## 项目简介

用 [Gymnasium](https://gymnasium.farama.org/) 学习强化学习，逐游戏实现多种算法，保留所有失败尝试。

**GitHub**: `LFrankl/RL-openai_gym`
**Git 作者**: `LFrankl <fuquanliang0@gmail.com>`

---

## 环境

| 项目 | 值 |
|------|----|
| Python | 3.11（`.python-version`）|
| 包管理 | uv（`pyproject.toml` + `uv.lock`）|
| 虚拟环境 | `.venv/` |
| torch | `<2.3`（macOS x86_64 兼容性限制）|

运行任何脚本前需激活 uv 环境：

```bash
uv run python <script.py>
# 或
source .venv/bin/activate && python <script.py>
```

---

## 目录结构

```
rl-gym/
├── cartpole/         Q-Learning / DQN / PPO
├── mountaincar/      Q-Learning / PPO / DQN+BC（稀疏奖励系列）
├── lunarlander/      DQN / Double DQN / Dueling DQN
├── pong/             CNN DQN（图像输入，Atari）
├── pyproject.toml
└── DEPLOY.md
```

每个子目录包含：
- 算法实现文件（`.py`）
- `THEORY.md`：严格理论分析文档

---

## 编码规范

### 新增算法文件

1. **MPS 注释**：所有含 `torch.device` 的文件，CPU 行下紧跟一行注释掉的 MPS 版本：

   ```python
   DEVICE = torch.device("cpu")
   # DEVICE = torch.device("mps")  # Mac M4 启用 GPU 加速（取消注释即可）
   ```

2. **评估函数**：必须包含 `render` 参数，默认 `True`，训练完自动播放动画：

   ```python
   def evaluate(model, episodes=10, render: bool = True):
       env = gym.make("GameName-v0", render_mode="human" if render else None)
   ```

3. **ALE 游戏**：文件顶部需注册环境：

   ```python
   import ale_py
   gym.register_envs(ale_py)
   ```

4. **注释风格**：中文注释，突出教学目的，说明每个设计决策的理由。

5. **失败实验**：保留所有失败版本，不删除。失败代码和成功代码同等重要。

### THEORY.md 规范

- 语言：中文
- 风格：严格理论分析，包含数学推导、定理、收敛性证明
- 结构：问题形式化 → 算法推导 → 失败分析（如有）→ 实验对比表 → 参考文献
- 公式：LaTeX 格式

---

## Git 规范

提交时指定作者信息：

```bash
git -c user.name="LFrankl" -c user.email="fuquanliang0@gmail.com" commit -m "..."
```

提交信息要求：
- 中文
- 去除 AI 味，口语自然
- 简短说明做了什么、为什么

---

## GitHub Pages 网站

项目网站位于 `docs/index.html`（单文件，无构建工具）。

**每次更新算法或内容时，同步更新网站：**

| 更新内容 | 需同步更新网站的位置 |
|---------|-------------------|
| 新增算法 | `#games` 区游戏卡片的算法表格（`.algo-table`）|
| 新增游戏 | `#games` 区新增 `.game-card` 块；`#algorithms` 对比表新增行；统计数字 |
| 评估分数变化 | 对应游戏卡片内的 `<td>` 分数 |
| 顶部统计数字 | `.stats-grid` 内的 4 个数字（环境数、算法数、失败实验数、文档数）|
| 新增失败实验 | `#journey` 时间线（MountainCar）或 `#philosophy` 的失败计数卡片 |
| 理论核心公式 | `#theory` 区的 `.theory-card` 块 |

**维护原则：**
- 网站是单文件 HTML（CSS/JS 全部内联），直接编辑 `docs/index.html`
- KaTeX 数学公式用 `$$...$$`（块级）或 `$...$`（行内）
- 与 README.md 保持一致，两者描述相同内容时保持数据同步
- 不要引入构建工具或额外文件依赖

---

## 已完成游戏一览

| 游戏 | 特点 | 算法 |
|------|------|------|
| CartPole-v1 | 奖励稠密，入门 | Q-Learning / DQN / PPO |
| MountainCar-v0 | 奖励极稀疏，需要探索 | Q-Learning(失败) / PPO(失败) / **BC(成功)** |
| LunarLander-v3 | 多目标稠密奖励 | DQN / Double DQN / **Dueling DQN(解决)** |
| ALE/Pong-v5 | 图像输入，Atari | CNN DQN + 帧堆叠 |

---

## 常见坑

- `torch<2.3` 与 `numpy>=2.x` 存在兼容警告（`_ARRAY_API not found`），不影响运行，无需处理
- mini-batch 索引用 `torch.randperm`，不用 numpy 数组索引 torch tensor
- ALE 游戏需要 `gym.register_envs(ale_py)`，否则报 `NamespaceNotFound`
- MountainCar DQN 微调阶段会出现灾难性遗忘，BC 预训练后不做 RL 微调
