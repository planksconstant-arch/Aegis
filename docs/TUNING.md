# Hyperparameter Tuning Guide

This document explains the key hyperparameters in Aegis's RL pipeline, when to adjust them, and what effects they have on training stability and agent performance.

---

## 1. Discount Factor (`gamma`)

| Setting | Default | Range |
|---------|---------|-------|
| `gamma` | `0.99`  | `0.9 – 0.999` |

**What it does:** Controls how much the agent values future rewards vs. immediate rewards.

**When to adjust:**
- **Lower (0.9–0.95):** Use when your tasks are short-horizon (e.g., single-file linting fixes). The agent will focus on immediate compilation success rather than speculative long-term benefits.
- **Higher (0.99–0.999):** Use when tasks span multiple steps (e.g., multi-file refactors where early edits only pay off after tests pass later).

---

## 2. PPO Clip Epsilon (`ppo_clip_epsilon`)

| Setting | Default | Range |
|---------|---------|-------|
| `ppo_clip_epsilon` | `0.2` | `0.1 – 0.3` |

**What it does:** Limits how much the policy can change in a single update step. Prevents catastrophic policy collapse.

**When to adjust:**
- **Lower (0.1):** Use when training is unstable (reward oscillates wildly). Tighter clipping = more conservative updates.
- **Higher (0.3):** Use when training is too slow to converge. Looser clipping = faster learning but riskier.

> [!TIP]
> If you see the agent "forgetting" good behaviors after a few hundred episodes, **lower** this value first.

---

## 3. Entropy Coefficient (`entropy_coefficient`)

| Setting | Default | Range |
|---------|---------|-------|
| `entropy_coefficient` | `0.01` | `0.001 – 0.05` |

**What it does:** Encourages the agent to explore diverse actions by adding an entropy bonus to the loss function.

**When to adjust:**
- **Lower (0.001):** When the agent has learned good behaviors and you want it to exploit them consistently.
- **Higher (0.03–0.05):** When the agent gets stuck repeating the same action (e.g., always choosing `no_op`). More entropy = more exploration.

> [!WARNING]
> Setting this above `0.1` will make the agent act almost randomly, defeating the purpose of RL training.

---

## 4. Critic Coefficient (`critic_coefficient`)

| Setting | Default | Range |
|---------|---------|-------|
| `critic_coefficient` | `0.5` | `0.25 – 1.0` |

**What it does:** Controls the relative weight of the critic (value function) loss vs. the actor (policy) loss.

**When to adjust:**
- **Lower (0.25):** If the critic is dominating training and the actor's policy isn't improving.
- **Higher (1.0):** If value estimates are wildly inaccurate (large TD errors), giving the critic more gradient signal to stabilize.

---

## 5. PER Alpha (`per_alpha`)

| Setting | Default | Range |
|---------|---------|-------|
| `per_alpha` | `0.6` | `0.0 – 1.0` |

**What it does:** Controls prioritization strength in the replay buffer. `0.0` = uniform sampling, `1.0` = fully prioritized.

**When to adjust:**
- **Lower (0.3–0.4):** If you notice the agent "overfitting" to a few high-error transitions and ignoring the rest.
- **Higher (0.7–0.9):** If learning is slow because the agent rarely revisits high-impact experiences.

---

## 6. PER Beta (`per_beta_start`)

| Setting | Default | Range |
|---------|---------|-------|
| `per_beta_start` | `0.4` | `0.0 – 1.0` |

**What it does:** Controls importance-sampling correction. Anneals linearly to `1.0` over `per_beta_steps`.

**When to adjust:**
- **Lower (0.2):** Acceptable early in training when some bias is tolerable for faster learning.
- Keep the annealing schedule (`per_beta_steps`) long enough that β reaches 1.0 before training ends.

---

## 7. Conservative Q-Learning Alpha (`cql_alpha`)

| Setting | Default | Range |
|---------|---------|-------|
| `cql_alpha` | `0.5` | `0.0 – 2.0` |

**What it does:** Regularizes Q-values to prevent overestimation of out-of-distribution actions. Critical for offline RL.

**When to adjust:**
- **Lower (0.1–0.3):** When you have plenty of online experience and don't need aggressive regularization.
- **Higher (1.0–2.0):** When bootstrapping from an offline dataset where many actions were never actually tried. Higher values penalize unseen actions more.

> [!IMPORTANT]
> If you're running purely online (no offline dataset), you can set this to `0.0` safely.

---

## 8. Curiosity Beta (`curiosity_beta`)

| Setting | Default | Range |
|---------|---------|-------|
| `curiosity_beta` | `0.005` | `0.001 – 0.05` |

**What it does:** Scales the intrinsic curiosity reward (from Random Network Distillation) before adding it to the environment reward.

**When to adjust:**
- **Lower (0.001):** When the environment reward signal is strong and you don't want curiosity to interfere.
- **Higher (0.01–0.05):** When the agent faces sparse rewards (e.g., tests only pass/fail with no partial credit) and needs exploration incentive.

> [!CAUTION]
> Setting this too high can make the agent "addicted to novelty"—constantly exploring without exploiting known-good strategies.

---

## 9. N-Step Returns (`n_step`)

| Setting | Default | Range |
|---------|---------|-------|
| `n_step` | `5` | `1 – 20` |

**What it does:** Number of lookahead steps for computing TD targets. Higher values propagate rewards faster but increase variance.

**When to adjust:**
- **Lower (1–3):** For more stable but slower learning. Use when reward signals are noisy.
- **Higher (10–20):** For faster credit assignment in long-horizon tasks. Use when you trust the environment rewards are clean.

---

## Quick Reference Table

| Symptom | Try Adjusting |
|---------|---------------|
| Agent keeps repeating the same action | ↑ `entropy_coefficient` to 0.03 |
| Training reward oscillates wildly | ↓ `ppo_clip_epsilon` to 0.1 |
| Agent ignores rare but important failures | ↑ `per_alpha` to 0.8 |
| Q-values explode to ±100+ | ↑ `cql_alpha` to 1.0 |
| Agent never tries new strategies | ↑ `curiosity_beta` to 0.02 |
| Very slow convergence | ↑ `learning_rate` to 1e-3, ↑ `n_step` to 10 |
| Agent "forgets" good behaviors | ↓ `ppo_clip_epsilon`, ↓ `learning_rate` |
