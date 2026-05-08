<h1 align="center">Local IDE Agent: A Hybrid LLM-RL Framework for Autonomous, Privacy-Preserving Code Generation</h1>

<p align="center">
  <b>A Pure-NumPy Actor-Critic implementation demonstrating localized continuous learning without cloud dependence.</b>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/planksconstant-arch/Aegis/actions"><img src="https://github.com/planksconstant-arch/Aegis/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
</p>

---

## Abstract

Modern AI coding assistants rely heavily on massive foundational models executing in cloud environments, leading to privacy concerns and a lack of user-specific personalization. We present the **Local IDE Agent**, an entirely offline, privacy-preserving hybrid framework. By coupling a localized Large Language Model (LLM) for candidate generation with a Pure-NumPy Twin-Q Actor-Critic Reinforcement Learning (RL) decision engine, the agent structurally learns human stylistic preferences over time. We solve standard sparse-reward limitations using automated Shadow Workspaces that supply dense compiler and linter rewards, achieving stable convergence via Random Network Distillation (RND) curiosity and N-step returns. 

---

## 1. Introduction

Developer environments currently face a dichotomy: utilize highly capable but privacy-invasive cloud LLMs, or attempt discrete pattern mapping using limited local mechanisms. The Local IDE Agent bridges this gap by providing an open-source, mathematically sound Reinforcement Learning apparatus that acts as a localized "Brain". 

The framework requires **no deep learning frameworks (e.g., PyTorch, TensorFlow)** for its RL matrix operations, ensuring extremely lightweight background execution. It treats the integrated development environment (IDE) as a Markov Decision Process (MDP), selectively querying local LLM inferences (e.g., via Ollama/vLLM) only when high-value patches are required, and mathematically ranking the resulting outputs based on learned personal preference.

---

## 2. Architecture & Methodology

Our framework decomposes into three interlocking sub-systems designed for stability in unbounded action spaces.

### 2.1 State Representation & Fusion
The IDE state $S_t$ is encoded via deterministic hashing or local Sentence-Transformer embeddings into three streams, fused by a **Learned Cross-Attention** layer into a 576-dimensional vector:
1. **Semantic Code Context** (384-d input → 256-d attended via $W_V$ projection): Open files, active diagnostics.
2. **Session Context** (64-d input → 256-d via $W_C$ projection): Task pressure, localized time, session phase.
3. **Behavioral Context** (128-d passthrough): Historical user acceptance latency, edit distance, revert rates.

The three streams are concatenated (640-d) and projected to 576-d through a learnable $W_{out}$ matrix. See `rl/attention.py` for the full Q/K/V derivation.

### 2.2 Hybrid LLM-RL Decision Engine
To resolve the infinite-action space inherent in open-ended text generation, we collapse the action space into a ranking paradigm:
- **LLM Candidate Generation**: An LLM generates a set $A = \{a_1, a_2, \ldots, a_K\}$ of candidate diffs.
- **Q-Value Ranking**: The Actor-Critic policy queries its Twin-Q networks to estimate the anticipated preference reward $Q(S_t, a_k)$ for each candidate.
- The framework guarantees monotonic improvement via a Clipped PPO-style policy update mapped onto the preferred actions. 

### 2.3 Automated Shadow Workspaces (Reward Densification)
Relying strictly on binary human approval creates a critically sparse reward signal. We introduce **Automated Shadow Evaluators**:
When candidates are generated, the agent clones the workspace into an isolated `.shadow_eval` background directory. It applies the respective diffs and subjects them to:
1. **Syntactic Validations:** Evaluated via native Python compiler abstraction.
2. **Static Analysis:** Evaluated via headless `Ruff` execution.
3. **Test Regression:** Evaluated via `Pytest` success rates. 

Candidates are given dense, shaped rewards before being presented to the final user.

---

## 3. Mathematical Stabilizations

Training Deep RL algorithms on highly non-stationary user feedback distributions requires careful mathematical regularizations. We implemented:

* **Twin-Q Target Networks:** Dual independent Q-predictors utilizing Polyak averaging ($\tau=0.005$) to safely bootstrap temporal-difference (TD) errors without catastrophic divergence.
* **Random Network Distillation (RND):** An intrinsic curiosity module delivering bonus rewards for exploring unvisited IDE states. Backward passes are structurally isolated to prevent activation cache corruption in the shared Multi-Layer Perceptron (MLP) trunk.
* **N-Step Returns ($n=5$):** Enhances credit assignment for delayed outcomes.
* **Welford Running Reward Normalization:** Prevents gradient explosion induced by unbound magnitude variance in intrinsic and extrinsic reward combinations.

---

## 4. Installation & Usage

### 4.1 Quick Start 
For rapid utilization, the system is distributed via standard Python tooling with absolutely minimal system dependencies.

```bash
# 1. Install via pip
pip install aegis-agent

# 2. Train the baseline policy (saves to .agent/policy_weights.npz)
aegis-agent train --episodes 50

# 3. Launch live telemetry monitoring
aegis-agent dashboard
```

Alternatively, use the provided one-click installers for OS-native fetching:
* **macOS / Linux:** `./install.sh`
* **Windows:** `.\install.ps1`

### 4.2 Connecting to your IDE (HTTP Bridge)
The agent operates as a headless daemon listening on `127.0.0.1:8765`. 

1. **Start the background process:**
   ```bash
   aegis-agent serve-bridge
   ```
2. **Telemetry Post (`/tick`):** Your IDE extension (VS Code, Neovim) submits the current code context and failing compiler diagnostics to the agent.
3. **Decision Return:** The agent returns a mathematically ranked `CandidatePatch` specifically formulated to pass local background tests.
4. **Reward Post (`/feedback`):** User decisions ("Accept/Reject") are posted back to the bridge, triggering a formal Bellman update to the system's neural matrices.

See [`settings.example.yaml`](settings.example.yaml) for instructions on integrating customized local LLMs (e.g. configuring `base_url` for Ollama).

---

## 5. Persistence & Privacy

This model is constrained exclusively to the host hardware. By default, interactions generate the `.agent/` directory within your workspace containing:
- `agent.db` (SQLite relational store for transitions)
- `policy_weights.npz` (Trained matrices)

**No external TCP connections** are made aside from those to explicitly configured local inference backends. You are advised to append both `.agent/` and `.shadow_eval/` to your `.gitignore`.

---

## 6. Citation & Contributing

> **Authors:** This architecture was constructed as an open-source initiative to democratize privacy-focused AI code generation.

We welcome contributions addressing extensions to Neovim LSP endpoints or VS Code extension deployments. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for pure-NumPy design governance guidelines and CI/CD validation strictures.
