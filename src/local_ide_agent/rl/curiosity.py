"""
Random Network Distillation (RND) — Intrinsic Curiosity Reward.

Solves the sparse-reward problem: in real IDE usage, the user may not
accept/reject suggestions for long stretches. Without an intrinsic reward
signal the policy barely trains. RND adds a bonus that rewards the agent
for visiting *novel* states, keeping the policy improving between rare
user feedback events.

Algorithm (Burda et al., 2018)
-------------------------------
  Fixed random target network:   f_target(s)  -> R^d        (never trained)
  Trained predictor network:     f_pred(s)    -> R^d        (trained to predict f_target)

  Intrinsic reward: r_int(s) = ||f_target(s) - f_pred(s)||^2

  As the predictor fits familiar states, the error (and thus bonus) shrinks
  → high bonus only for genuinely novel / rare observations.

Integration
-----------
  combined_reward = r_ext + beta_curiosity * r_int

  r_ext comes from the environment / user feedback.
  r_int is computed here for each new state before storing in the replay buffer.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from local_ide_agent.rl.nn import AdamOptimizer, MLP, load_weights, save_weights


class RNDModule:
    """
    Random Network Distillation curiosity module.

    Parameters
    ----------
    state_dim:    dimension of the input state vector (576 by default)
    embed_dim:    dimension of the RND embedding (64-d is fast and sufficient)
    learning_rate: learning rate for training the predictor
    beta:         weight on the intrinsic reward term
    normalize:    whether to normalise intrinsic rewards by running std
    weight_path:  where to save/load predictor weights
    """

    def __init__(
        self,
        state_dim: int = 576,
        embed_dim: int = 64,
        learning_rate: float = 1e-3,
        beta: float = 0.1,
        normalize: bool = True,
        weight_path: str | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.beta = beta
        self.normalize = normalize
        self.weight_path = weight_path

        # Fixed random target (never updated)
        self.target_net = MLP([state_dim, 128, embed_dim], name="rnd_target")

        # Trained predictor — give each layer its own Adam optimizer
        self.predictor_net = MLP([state_dim, 128, embed_dim], name="rnd_pred")
        self.predictor_opts: list[AdamOptimizer] = [
            AdamOptimizer(lr=learning_rate, grad_clip=1.0)
            for _ in self.predictor_net.layers
        ]

        # Running statistics for reward normalisation
        self._reward_running_mean: float = 0.0
        self._reward_running_var: float = 1.0
        self._reward_count: int = 0
        self._update_count: int = 0

        self._load_weights()

    # ------------------------------------------------------------------
    # Intrinsic reward computation
    # ------------------------------------------------------------------

    def intrinsic_reward(self, state_vector: list[float] | np.ndarray) -> float:
        """
        Compute the intrinsic curiosity reward for a given state vector.
        Returns a non-negative scalar.
        """
        sv = np.asarray(state_vector, dtype=np.float64)

        target_out = self.target_net.forward(sv)
        pred_out = self.predictor_net.forward(sv)

        error = pred_out - target_out
        r_int = float(np.sum(error ** 2))

        if self.normalize:
            r_int = self._normalize_reward(r_int)

        return self.beta * max(0.0, r_int)

    def _normalize_reward(self, r: float) -> float:
        """Online Welford running mean/variance normalisation."""
        self._reward_count += 1
        n = self._reward_count
        old_mean = self._reward_running_mean
        self._reward_running_mean += (r - old_mean) / n
        self._reward_running_var += (r - old_mean) * (r - self._reward_running_mean)
        std = math.sqrt(self._reward_running_var / max(n, 1)) + 1e-8
        return r / std

    # ------------------------------------------------------------------
    # Predictor training step
    # ------------------------------------------------------------------

    def _target_forward_no_cache(self, sv: np.ndarray) -> np.ndarray:
        """
        Run the target network forward pass using numpy directly,
        without touching the MLP activation cache (so curiosity_reward()
        and update() can both call the target without cache conflicts).
        """
        from local_ide_agent.rl.nn import relu
        h = sv.copy()
        # Apply layer norm (same as MLP does internally)
        mean = h.mean()
        std = h.std() + 1e-6
        h = (h - mean) / std
        for idx, layer in enumerate(self.target_net.layers):
            h = h @ layer.W + layer.b
            if idx < len(self.target_net.layers) - 1:
                h = relu(h)
        return h


    def update_and_get_reward(self, state_vector: list[float] | np.ndarray) -> float:
        """
        Atomically compute intrinsic reward AND update predictor.
        Uses a self-contained forward+backward that does NOT rely on
        MLP._pre_activations (which can be corrupted by concurrent policy
        forward passes sharing the numpy thread).
        Returns the intrinsic reward (beta-scaled and normalised).
        """
        sv = np.asarray(state_vector, dtype=np.float64)
        target_out = self._target_forward_no_cache(sv)
        raw_loss, r_int = self._predictor_update_inline(sv, target_out)
        if self.normalize:
            r_int = self._normalize_reward(r_int)
        return self.beta * max(0.0, r_int)

    def update(self, state_vector: list[float] | np.ndarray) -> float:
        """Train predictor one step; returns raw prediction error."""
        sv = np.asarray(state_vector, dtype=np.float64)
        target_out = self._target_forward_no_cache(sv)
        raw_loss, _ = self._predictor_update_inline(sv, target_out)
        return raw_loss

    def _predictor_update_inline(
        self,
        sv: np.ndarray,
        target_out: np.ndarray,
    ) -> tuple[float, float]:
        """
        Full forward + backward + parameter update for the predictor,
        computed INLINE without storing to MLP._pre_activations.

        Returns (raw_loss, normalised_loss).
        No MLP.forward() / MLP.backward() is called — we use the Linear
        layers' own .forward() / .backward() directly so each layer's
        _last_input cache is local to this call.
        """
        from local_ide_agent.rl.nn import relu, relu_grad

        layers = self.predictor_net.layers
        opts = self.predictor_opts

        # ---- Forward (layer norm then each Linear+relu) ----
        h = sv.copy()
        mean = h.mean(); std = h.std() + 1e-6
        h = (h - mean) / std  # layer norm

        pre_acts: list[np.ndarray] = []
        for idx, layer in enumerate(layers):
            pre = layer.forward(h)   # stores h in layer._last_input ✓
            pre_acts.append(pre)
            h = relu(pre) if idx < len(layers) - 1 else pre

        pred_out = h
        error = pred_out - target_out
        raw_loss = float(np.sum(error ** 2))

        # ---- Backward (MSE gradient: 2*(pred-target)) ----
        g = 2.0 * error
        for idx in reversed(range(len(layers))):
            if idx < len(layers) - 1:
                g = g * relu_grad(pre_acts[idx])
            grad_x, gW, gb = layers[idx].backward(g)
            layers[idx].apply_gradients(gW, gb, opts[idx])
            g = grad_x  # propagate to previous layer

        self._update_count += 1
        if self._update_count % 256 == 0 and self.weight_path:
            self._save_weights()

        return raw_loss, raw_loss


    def update_batch(self, state_vectors: list[list[float]]) -> float:
        """Train predictor on a batch; returns mean intrinsic reward."""
        if not state_vectors:
            return 0.0
        rewards = [self.update_and_get_reward(sv) for sv in state_vectors]
        return sum(rewards) / len(rewards)


    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def _save_weights(self) -> None:
        if not self.weight_path:
            return
        try:
            w = self.predictor_net.get_weights()
            prefixed = {f"rnd_pred_{k}": v for k, v in w.items()}
            save_weights(prefixed, self.weight_path)
        except Exception:
            pass

    def _load_weights(self) -> None:
        if not self.weight_path:
            return
        try:
            stored = load_weights(self.weight_path)
            if not stored:
                return
            unprefixed = {
                k.replace("rnd_pred_", "", 1): v
                for k, v in stored.items()
                if k.startswith("rnd_pred_")
            }
            self.predictor_net.set_weights(unprefixed)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stats for logging
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        return {
            "rnd_updates": float(self._update_count),
            "rnd_reward_mean": round(self._reward_running_mean, 5),
            "rnd_reward_std": round(
                math.sqrt(self._reward_running_var / max(self._reward_count, 1)), 5
            ),
            "rnd_beta": self.beta,
        }
