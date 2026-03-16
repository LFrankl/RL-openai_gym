# 部署指南

依赖全部锁定在 `uv.lock`，用 uv 的虚拟环境跑，不会碰系统 Python。

## 前置条件

- Python 3.11+（系统自带即可）
- 网络能访问 PyPI（torch 约 140MB）

## 步骤

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # 让当前 shell 认识 uv
```

验证：

```bash
uv --version
```

### 2. 拉代码

```bash
git clone <仓库地址>
cd rl-gym
```

### 3. 建虚拟环境 + 装依赖

```bash
uv sync
```

`uv sync` 会读 `uv.lock`，版本完全对齐，不会影响系统任何包。
虚拟环境建在项目内的 `.venv/`，删掉目录就能完全清理。

### 4. 跑示例

```bash
# Q-Learning
uv run python cartpole/q_learning.py

# DQN
uv run python cartpole/dqn.py
```

用 `uv run` 不需要手动激活虚拟环境。

---

## 清理

```bash
rm -rf .venv
```

系统环境完全不受影响。

---

## 常见问题

**torch 下载超时**

换镜像源：

```bash
uv sync --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
```

**numpy 版本警告（`_ARRAY_API not found`）**

torch 2.2 与 numpy 2.x 有兼容问题，警告不影响运行。
等网络畅通后执行 `uv add "numpy<2"` 可消除。

**macOS 无法安装最新 torch**

torch 2.3+ 不提供 macOS x86_64 的 wheel，项目已锁定 `torch<2.3`，`uv sync` 会自动选对版本。
