# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-04-22

### Added — Advanced RL Primitives
- **Random Network Distillation (RND)** curiosity module (`rl/curiosity.py`)
  — intrinsic reward for novel states, prevents policy collapse during sparse-reward periods
- **N-step return buffer** (`rl/n_step.py`, n=5) — 5× faster credit propagation vs TD(0)
- **Curriculum learning scheduler** (`training/curriculum.py`)
  — auto-promotes EASY → MEDIUM → HARD based on rolling success rate window
- **Trajectory buffer** (`agent/trajectory_buffer.py`)
  — 10-step episode memory injected into every observation for short-term context
- **Evaluation harness** (`rl/eval.py`)
  — 5 held-out tasks, measures generalization gap (train reward vs eval reward)

### Added — Training Stability
- **Target Q-networks** (`rl/policy.py`)
  — EMA-updated shadow copies of Q1/Q2 (Polyak τ=0.005); eliminates deadly triad instability
- **Soft action masking** — additive logit penalties for inappropriate actions (e.g. `no_op` when diagnostics exist)
- **Reward normalisation** (`training/loop.py`)
  — Welford running mean/std, prevents intrinsic reward dominating over environment reward
- **Gradient clipping** (`rl/nn.py`) — AdamOptimizer clips gradients to prevent explosion

### Added — Observability
- **Live terminal dashboard** (`cli/dashboard.py`, `local-ide-agent dashboard`)
  — ANSI sparkline of recent rewards, TD error, action success rates, weight file status
- **GET /training-status bridge endpoint** — exposes epsilon, buffer fill, reward history, action rates
- **Incremental action_success_rates table** (`memory/store.py`)
  — materialised view updated O(1) on each feedback write; replaces O(n) snapshot scan

### Added — Real-World Grounding
- **Real workspace context injection** (`training/environment.py`)
  — `configure_workspace()` reads actual `.py` files and `events.jsonl` diagnostics into episodes
- **`local-ide-agent eval`** CLI command — runs policy against held-out tasks with generalization gap

### Added — Config
- New `rl.*` config fields: `curiosity_*`, `n_step`, `curriculum_*`, `trajectory_window`

### Changed
- `training/loop.py` fully refactored — wires all new components with per-episode observability logging
- `settings.example.yaml` updated with real values and inline comments

---

## [0.1.0] — 2026-04-07

### Added — Core Scaffold
- Actor-Critic policy with shared MLP trunk + twin-Q heads (`rl/policy.py`)
- Pure-numpy MLP, LayerNorm, Adam optimiser, Huber loss (`rl/nn.py`)
- Prioritized Experience Replay buffer with IS-weights (`rl/replay.py`)
- Cross-attention state fusion: code + behaviour + session context → 576-d vector (`rl/state.py`, `rl/attention.py`)
- GAE advantages, PPO-clip policy gradient, entropy regularisation (`rl/trainer.py`)
- Simulated multi-step coding environment with 22 task templates (`training/environment.py`)
- Local HTTP bridge server with `/tick`, `/feedback`, `/memory`, `/training-status` (`bridge.py`)
- SQLite memory store: feedback, replay transitions, episode logs, style preferences (`memory/store.py`)
- Shadow workspace manager for safe autonomous experimentation (`shadow/workspace.py`)
- Background deployment manager with heartbeat (`deployment/background.py`)
- Autonomy gating: auto-low-risk / require-approval / block-high-risk (`agent/core.py`)
- Local IDE connector: file read/write, bounded command execution, event logging (`connectors/ide.py`)
- Full CLI: `train`, `run`, `serve-bridge`, `shadow-run`, `counterfactual-lab`, `research-plan`
- CQL offline regulariser — *planned but not yet implemented*; config stub removed in v0.3.0
