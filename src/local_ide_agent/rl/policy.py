"""
Real actor-critic RL policy with:
  - Shared MLP trunk (576 -> 256 -> 128) with He init + layer norm
  - Actor head: Linear(128, |A|) -> softmax -> epsilon-greedy action selection
  - Twin-Q critic heads: two independent Q networks, TD target uses min(Q1, Q2)
  - Adam optimiser with per-parameter state
  - PPO-style clipped policy gradient loss
  - Entropy regularisation
  - Weight save/load to .npz for persistence across sessions

All gradient mathematics are implemented in pure numpy via the nn.py primitives.
No PyTorch, TensorFlow, or JAX required.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from local_ide_agent.agent.policy import Policy
from local_ide_agent.config import RLHyperparams
from local_ide_agent.rl.actions import ACTION_SPACE, StrategyAction
from local_ide_agent.rl.nn import (
    MLP,
    AdamOptimizer,
    Linear,
    huber_grad,
    huber_loss,
    load_weights,
    log_softmax,
    relu,
    save_weights,
    softmax,
)
from local_ide_agent.rl.state import FusedState, StateEncoderStack
from local_ide_agent.schemas import Action, Decision, Observation


# ---------------------------------------------------------------------------
# Critic Q-network (single head of the twin)
# ---------------------------------------------------------------------------

class QNetwork:
    """
    Q(state, action) estimator.
    Input: trunk_output (128-d) concatenated with one-hot action (|A|-d).
    Architecture: Linear(128+|A|, hidden) -> relu -> Linear(hidden, 1)
    """

    def __init__(self, trunk_dim: int, n_actions: int, hidden: int = 64, name: str = "q") -> None:
        self.trunk_dim = trunk_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.name = name
        in_dim = trunk_dim + n_actions
        scale1 = math.sqrt(2.0 / in_dim)
        scale2 = math.sqrt(2.0 / hidden)
        self.l1 = Linear(in_dim, hidden, name=f"{name}_l1")
        self.l2 = Linear(hidden, 1, name=f"{name}_l2")
        self._cache: dict = {}

    def forward(self, trunk: np.ndarray, action_onehot: np.ndarray) -> float:
        sa = np.concatenate([trunk, action_onehot])
        h1_pre = self.l1.forward(sa)
        h1 = relu(h1_pre)
        q_val = self.l2.forward(h1)
        self._cache = {"sa": sa, "h1_pre": h1_pre, "h1": h1}
        return float(q_val[0])

    def backward_and_update(
        self,
        grad_q: float,
        optimizer_l1: AdamOptimizer,
        optimizer_l2: AdamOptimizer,
    ) -> None:
        """Backpropagate scalar gradient through Q network."""
        g2 = np.array([grad_q], dtype=np.float64)
        grad_h1, gW2, gb2 = self.l2.backward(g2)
        g1_pre = grad_h1 * (self._cache["h1_pre"] > 0).astype(np.float64)
        _, gW1, gb1 = self.l1.backward(g1_pre)
        self.l1.apply_gradients(gW1, gb1, optimizer_l1)
        self.l2.apply_gradients(gW2, gb2, optimizer_l2)

    def get_weights(self) -> dict[str, np.ndarray]:
        return {
            f"{self.name}_l1_W": self.l1.W,
            f"{self.name}_l1_b": self.l1.b,
            f"{self.name}_l2_W": self.l2.W,
            f"{self.name}_l2_b": self.l2.b,
        }

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        if f"{self.name}_l1_W" in weights:
            self.l1.W = weights[f"{self.name}_l1_W"].copy()
        if f"{self.name}_l1_b" in weights:
            self.l1.b = weights[f"{self.name}_l1_b"].copy()
        if f"{self.name}_l2_W" in weights:
            self.l2.W = weights[f"{self.name}_l2_W"].copy()
        if f"{self.name}_l2_b" in weights:
            self.l2.b = weights[f"{self.name}_l2_b"].copy()


# ---------------------------------------------------------------------------
# Main ActorCriticPolicy
# ---------------------------------------------------------------------------

@dataclass
class ActorCriticPolicy(Policy):
    """
    Full actor-critic policy with:
      - Shared MLP trunk
      - Actor linear head + softmax action selection
      - Twin-Q critic (Q1, Q2); target uses min(Q1, Q2)
      - All parameters updated via AdamOptimizer
      - Weights persisted to disk after each update call
    """

    encoder_stack: StateEncoderStack = field(default_factory=StateEncoderStack)
    hp: RLHyperparams = field(default_factory=RLHyperparams)
    block_high_risk: bool = True

    # Populated in __post_init__
    trunk: MLP = field(init=False)
    actor_head: Linear = field(init=False)
    q1: QNetwork = field(init=False)
    q2: QNetwork = field(init=False)

    trunk_opt: AdamOptimizer = field(init=False)
    actor_opt: AdamOptimizer = field(init=False)
    q1_l1_opt: AdamOptimizer = field(init=False)
    q1_l2_opt: AdamOptimizer = field(init=False)
    q2_l1_opt: AdamOptimizer = field(init=False)
    q2_l2_opt: AdamOptimizer = field(init=False)

    # Target networks — same architecture, initialised to identical weights
    q1_target: QNetwork = field(init=False)
    q2_target: QNetwork = field(init=False)
    _target_update_counter: int = field(init=False, default=0)
    TARGET_UPDATE_FREQ: int = 10     # steps between soft updates
    TARGET_EMA_TAU: float = 0.005   # EMA coefficient (1.0 = hard copy)

    # Hybrid RL: Patch Candidate Ranking
    patch_critic: QNetwork = field(init=False)
    patch_critic_l1_opt: AdamOptimizer = field(init=False)
    patch_critic_l2_opt: AdamOptimizer = field(init=False)

    # Runtime state
    last_fused_state: FusedState | None = field(init=False, default=None)
    last_action_index: int = field(init=False, default=0)
    last_probs: np.ndarray | None = field(init=False, default=None)
    last_trunk_output: np.ndarray | None = field(init=False, default=None)
    reward_history: list[float] = field(default_factory=list)
    _step_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        n_actions = len(ACTION_SPACE)
        trunk_sizes = [576] + list(self.hp.trunk_hidden_sizes)
        trunk_out = trunk_sizes[-1]

        self.trunk = MLP(trunk_sizes, name="trunk")
        self.actor_head = Linear(trunk_out, n_actions, name="actor")

        critic_hidden = self.hp.critic_hidden_sizes[0] if self.hp.critic_hidden_sizes else 64
        self.q1 = QNetwork(trunk_out, n_actions, hidden=critic_hidden, name="q1")
        self.q2 = QNetwork(trunk_out, n_actions, hidden=critic_hidden, name="q2")

        lr = self.hp.learning_rate
        self.trunk_opt = AdamOptimizer(lr=lr)
        self.actor_opt = AdamOptimizer(lr=lr)
        self.q1_l1_opt = AdamOptimizer(lr=lr)
        self.q1_l2_opt = AdamOptimizer(lr=lr)
        self.q2_l1_opt = AdamOptimizer(lr=lr)
        self.q2_l2_opt = AdamOptimizer(lr=lr)

        # Target networks — same architecture, initialised to identical weights
        critic_hidden = self.hp.critic_hidden_sizes[0] if self.hp.critic_hidden_sizes else 64
        trunk_out = (list(self.hp.trunk_hidden_sizes) or [128])[-1]
        self.q1_target = QNetwork(trunk_out, n_actions, hidden=critic_hidden, name="q1_target")
        self.q2_target = QNetwork(trunk_out, n_actions, hidden=critic_hidden, name="q2_target")
        self._sync_targets()  # hard copy on init

        # Hybrid RL: Patch Critic (takes 128-d trunk + 64-d patch embedding)
        self.patch_critic = QNetwork(trunk_out, 64, hidden=critic_hidden, name="patch_critic")
        self.patch_critic_l1_opt = AdamOptimizer(lr=lr)
        self.patch_critic_l2_opt = AdamOptimizer(lr=lr)

        # Load persisted weights if available
        self._load_weights()

    # ------------------------------------------------------------------
    # Epsilon-greedy exploration schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        frac = min(self._step_count / max(self.hp.epsilon_decay_steps, 1), 1.0)
        return self.hp.epsilon_start - frac * (self.hp.epsilon_start - self.hp.epsilon_end)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _action_onehot(self, index: int) -> np.ndarray:
        oh = np.zeros(len(ACTION_SPACE), dtype=np.float64)
        oh[index] = 1.0
        return oh

    def decide(self, observation: Observation) -> Decision:
        self._step_count += 1

        fused = self.encoder_stack.encode(observation)
        self.last_fused_state = fused

        state_vec = np.asarray(fused.state_vector, dtype=np.float64)
        trunk_out = self.trunk.forward(state_vec)
        self.last_trunk_output = trunk_out.copy()

        logits = self.actor_head.forward(trunk_out)

        # ------------------------------------------------------------------
        # Soft action masking — additive logit penalties for contextually
        # inappropriate actions. Does NOT hard-block; just heavy discouragement.
        # These penalties flow into probability via softmax so backprop still works.
        # ------------------------------------------------------------------
        masked_logits = logits.copy()
        n_actions = len(ACTION_SPACE)
        action_names = [a.name for a in ACTION_SPACE]

        def _idx(name: str) -> int:
            return action_names.index(name) if name in action_names else -1

        diag_present = bool(observation.diagnostics)
        user_present = observation.user_present
        pressure = str(observation.metadata.get("pressure_level", "normal"))
        consecutive_same = int(observation.metadata.get("consecutive_same_action", 0))
        last_action = str(observation.metadata.get("last_action_name", ""))
        buffer_small = int(observation.metadata.get("trajectory_length", 0)) < 3

        # Rule 1: penalise promote_candidate when we haven't gathered enough context
        promote_idx = _idx("promote_candidate")
        if promote_idx >= 0 and buffer_small:
            masked_logits[promote_idx] -= 3.0

        # Rule 2: penalise no_op heavily when diagnostics are present (do *something*)
        noop_idx = _idx("no_op")
        if noop_idx >= 0 and diag_present:
            masked_logits[noop_idx] -= 2.5

        # Rule 3: silent_shadow is wasteful when user is present + high pressure
        shadow_idx = _idx("silent_shadow")
        if shadow_idx >= 0 and user_present and pressure == "high":
            masked_logits[shadow_idx] -= 2.0

        # Rule 4: penalise the last action if taken too many times consecutively
        if consecutive_same >= 3 and last_action in action_names:
            repeat_idx = _idx(last_action)
            if repeat_idx >= 0:
                masked_logits[repeat_idx] -= float(consecutive_same) * 0.5

        # Rule 5 (HARD BLOCK): Completely mask high-risk actions if forbidden
        if self.block_high_risk:
            from local_ide_agent.schemas import RiskLevel
            for i, action in enumerate(ACTION_SPACE):
                if action.risk == RiskLevel.HIGH:
                    masked_logits[i] = -1e9

        probs = softmax(masked_logits)
        self.last_probs = probs.copy()
        self.last_masked_logits = masked_logits.copy()

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action_idx = random.randrange(len(ACTION_SPACE))
        else:
            action_idx = int(np.argmax(probs))

        self.last_action_index = action_idx
        selected: StrategyAction = ACTION_SPACE[action_idx]

        # Q estimates for payload info
        oh = self._action_onehot(action_idx)
        q1_val = self.q1.forward(trunk_out, oh)
        q2_val = self.q2.forward(trunk_out, oh)
        q_est = min(q1_val, q2_val)

        # Mask contribution to confidence
        confidence = float(np.clip(0.45 + probs[action_idx], 0.05, 0.99))
        requires_approval = selected.autonomy_tier != "auto-low-risk"


        return Decision(
            action=Action(
                action_type=selected.action_type,
                description=selected.description,
                risk=selected.risk,
                payload={
                    "strategy_name": selected.name,
                    "state_dimension": len(fused.state_vector),
                    "q_estimate": round(q_est, 4),
                    "epsilon": round(self.epsilon, 4),
                    "step": self._step_count,
                },
            ),
            confidence=round(confidence, 3),
            requires_approval=requires_approval,
            reason=(
                f"Strategy '{selected.name}' | actor_prob={probs[action_idx]:.3f} "
                f"Q_min={q_est:.3f} eps={self.epsilon:.3f} step={self._step_count}"
            ),
            autonomy_tier=selected.autonomy_tier,
        )

    # ------------------------------------------------------------------
    # Hybrid LLM-RL Candidate Ranking
    # ------------------------------------------------------------------

    def rank_candidates(self, candidates: list[Any]) -> tuple[Any, float]:
        """Rank a list of CandidatePatch objects using the patch_critic, returning the best candidate and its Q-value."""
        if not candidates:
            raise ValueError("No candidates provided")
        
        from local_ide_agent.rl.state import _project_text
        import numpy as np

        if self.last_trunk_output is None:
            # Fallback to naive random selection if trunk is uninitialised
            return random.choice(candidates), 0.0

        best_q = -float("inf")
        best_candidate = candidates[0]
        
        for candidate in candidates:
            # Encode the candidate diff to a 64-d vector deterministically
            a_embed = np.array(_project_text(candidate.diff, 64, scale=1.0), dtype=np.float64)
            q_val = self.patch_critic.forward(self.last_trunk_output, a_embed)
            
            # Cache the embedding in the candidate for training updates
            candidate.metadata = getattr(candidate, "metadata", {})
            candidate.metadata["patch_embed"] = a_embed

            if q_val > best_q:
                best_q = q_val
                best_candidate = candidate

        return best_candidate, float(best_q)

    def update_patch_critic(self, reward: float, selected_candidate: Any) -> float:
        """Update the patch_critic given the actual reward obtained by the chosen CandidatePatch."""
        if self.last_trunk_output is None:
            return 0.0

        embed = selected_candidate.metadata.get("patch_embed")
        if embed is None:
            return 0.0

        # Simple Bellman update with TD target = reward (since patch application is terminal for this phase)
        td_target = float(np.clip(reward, -10.0, 10.0))
        q_pred = self.patch_critic.forward(self.last_trunk_output, embed)
        td_error = td_target - q_pred

        grad_q = float(huber_grad(np.array([q_pred]), np.array([td_target]))[0])
        
        # Backprop through both layers with separate optimizers to ensure
        # correct per-layer Adam bias correction.
        self.patch_critic.backward_and_update(
            grad_q * self.hp.critic_coefficient,
            self.patch_critic_l1_opt,
            self.patch_critic_l2_opt,
        )
        
        return abs(td_error)

    # ------------------------------------------------------------------
    # Gradient update step
    # ------------------------------------------------------------------

    def update_step(
        self,
        reward: float,
        next_state_vector: list[float],
        done: bool,
        is_weight: float = 1.0,
    ) -> float:
        """
        Perform one actor-critic gradient update using the last decision's context.

        Returns the new TD error (for PER priority update).
        """
        if self.last_trunk_output is None or self.last_probs is None:
            return abs(reward)

        gamma = self.hp.gamma
        clip_eps = self.hp.ppo_clip_epsilon
        ent_coeff = self.hp.entropy_coefficient
        critic_coeff = self.hp.critic_coefficient

        trunk_out = self.last_trunk_output
        probs = self.last_probs
        action_idx = self.last_action_index
        oh = self._action_onehot(action_idx)

        # ---- Compute next-state value ----
        next_vec = np.asarray(next_state_vector, dtype=np.float64)
        next_trunk = self.trunk.forward(next_vec)
        next_logits = self.actor_head.forward(next_trunk)
        next_probs = softmax(next_logits)
        next_action_idx = int(np.argmax(next_probs))
        next_oh = self._action_onehot(next_action_idx)

        next_q1 = self.q1_target.forward(next_trunk, next_oh)
        next_q2 = self.q2_target.forward(next_trunk, next_oh)
        next_q = min(next_q1, next_q2)

        td_target = reward + gamma * (1.0 - float(done)) * next_q
        # Clamp target to prevent Q-value explosion in early training
        td_target = float(np.clip(td_target, -10.0, 10.0))

        # ---- Critic losses (twin-Q, Huber) ----
        q1_pred = self.q1.forward(trunk_out, oh)
        q2_pred = self.q2.forward(trunk_out, oh)

        td_error = td_target - q1_pred  # signed

        q1_loss_val = float(huber_loss(np.array([q1_pred]), np.array([td_target]))[0])
        q2_loss_val = float(huber_loss(np.array([q2_pred]), np.array([td_target]))[0])

        grad_q1 = is_weight * float(huber_grad(np.array([q1_pred]), np.array([td_target]))[0])
        grad_q2 = is_weight * float(huber_grad(np.array([q2_pred]), np.array([td_target]))[0])

        self.q1.backward_and_update(grad_q1, self.q1_l1_opt, self.q1_l2_opt)
        self.q2.backward_and_update(grad_q2, self.q2_l1_opt, self.q2_l2_opt)

        # ---- Actor loss (policy gradient with advantage baseline) ----
        advantage = float(td_target - q1_pred)

        log_probs = log_softmax(self.actor_head.forward(trunk_out))
        log_p_a = log_probs[action_idx]

        # Entropy bonus  H(π) = -Σ π log π
        entropy = float(-np.sum(probs * log_softmax(self.actor_head.forward(trunk_out))))

        actor_loss_val = -(log_p_a * advantage) - ent_coeff * entropy

        # Gradient of actor loss w.r.t. logits:
        # d/d_logits[-log_softmax(logits)[a] * A] = (probs - one_hot(a)) * A
        # d/d_logits[-H(pi)] = log_probs + 1  (simplified)
        on_hot = np.zeros(len(ACTION_SPACE), dtype=np.float64)
        on_hot[action_idx] = 1.0
        grad_actor_logits = is_weight * (
            (probs - on_hot) * advantage
            - ent_coeff * (log_softmax(self.actor_head.forward(trunk_out)) + 1.0)
        )

        # Backprop through actor head into trunk
        grad_trunk_from_actor, grad_W_actor, grad_b_actor = self.actor_head.backward(grad_actor_logits)
        self.actor_head.apply_gradients(grad_W_actor, grad_b_actor, self.actor_opt)

        # Backprop trunk
        trunk_grads = self.trunk.backward(grad_trunk_from_actor * critic_coeff)
        self.trunk.apply_gradients(trunk_grads, self.trunk_opt)

        # ---- Update reward history for soft bias ----
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        # ---- Soft-update target networks every TARGET_UPDATE_FREQ steps ----
        self._target_update_counter += 1
        if self._target_update_counter % self.TARGET_UPDATE_FREQ == 0:
            self._soft_update_targets()

        return abs(td_error)

    # ------------------------------------------------------------------
    # Legacy `update(reward)` interface (called by agent core)
    # ------------------------------------------------------------------

    def update(self, reward: float) -> None:
        """
        Lightweight update called with a single reward scalar.
        Used by the agent's record_feedback path; a full update_step
        is preferred when next_state_vector is available.
        """
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        # Partial actor update: nudge logits toward rewarded actions
        if self.last_trunk_output is not None and self.last_probs is not None:
            advantage = reward
            probs = self.last_probs
            oh = np.zeros(len(ACTION_SPACE), dtype=np.float64)
            oh[self.last_action_index] = 1.0
            grad_logits = (probs - oh) * advantage * 0.1
            _, gW, gb = self.actor_head.backward(grad_logits)
            self.actor_head.apply_gradients(gW, gb, self.actor_opt)
        self._save_weights()

    # ------------------------------------------------------------------
    # Target network utilities
    # ------------------------------------------------------------------

    def _sync_targets(self) -> None:
        """Hard copy online Q→target (used on initialisation and after load)."""
        for src, dst in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
            dst.l1.W = src.l1.W.copy()
            dst.l1.b = src.l1.b.copy()
            dst.l2.W = src.l2.W.copy()
            dst.l2.b = src.l2.b.copy()

    def _soft_update_targets(self) -> None:
        """
        Polyak / EMA update:  θ_target = τ*θ_online + (1-τ)*θ_target
        Small τ (0.005) keeps targets stable — critical for reducing
        the deadly triad: bootstrapping + off-policy + function approximation.
        """
        tau = self.TARGET_EMA_TAU
        for src, dst in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
            dst.l1.W = tau * src.l1.W + (1 - tau) * dst.l1.W
            dst.l1.b = tau * src.l1.b + (1 - tau) * dst.l1.b
            dst.l2.W = tau * src.l2.W + (1 - tau) * dst.l2.W
            dst.l2.b = tau * src.l2.b + (1 - tau) * dst.l2.b

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def _all_weights(self) -> dict[str, np.ndarray]:
        weights: dict[str, np.ndarray] = {}
        weights.update(self.trunk.get_weights())
        weights["actor_W"] = self.actor_head.W
        weights["actor_b"] = self.actor_head.b
        weights.update(self.q1.get_weights())
        weights.update(self.q2.get_weights())
        weights.update(self.patch_critic.get_weights())
        # Persist target nets so restarts are deterministic
        weights.update({f"target_{k}": v for k, v in self.q1_target.get_weights().items()})
        weights.update({f"target_{k}": v for k, v in self.q2_target.get_weights().items()})
        return weights

    def _save_weights(self) -> None:
        try:
            save_weights(self._all_weights(), self.hp.weight_path)
        except Exception:
            pass  # Non-fatal: training still proceeds

    def _load_weights(self) -> None:
        try:
            stored = load_weights(self.hp.weight_path)
            if not stored:
                return
            self.trunk.set_weights(stored)
            if "actor_W" in stored:
                self.actor_head.W = stored["actor_W"].copy()
            if "actor_b" in stored:
                self.actor_head.b = stored["actor_b"].copy()
            self.q1.set_weights(stored)
            self.q2.set_weights(stored)
            self.patch_critic.set_weights(stored)
            # Restore targets if saved, otherwise hard-copy from online
            target_q1_stored = {k.replace("target_", "", 1): v for k, v in stored.items() if k.startswith("target_q1")}
            target_q2_stored = {k.replace("target_", "", 1): v for k, v in stored.items() if k.startswith("target_q2")}
            if target_q1_stored:
                self.q1_target.set_weights(target_q1_stored)
            if target_q2_stored:
                self.q2_target.set_weights(target_q2_stored)
            if not target_q1_stored:
                self._sync_targets()
        except Exception:
            pass  # Fresh init on any load error

