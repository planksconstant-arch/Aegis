"""
Learned cross-attention fusion layer.

Replaces the fixed weighted sum (0.5*code + 0.3*behavior + 0.2*context)
with a proper query/key/value attention mechanism whose projection matrices
are updated by gradient descent.

Architecture
------------
  Q = behavior_embed @ W_Q          # (d_behavior -> d_k)
  K = code_embed    @ W_K          # (d_code    -> d_k)
  V = code_embed    @ W_V          # (d_code    -> d_v)
  context_proj = context_embed @ W_C  # (d_ctx -> d_v)

  attn_weights = softmax(Q @ K.T / sqrt(d_k))  # scalar for 1-d case
  attended = attn_weights * V
  fused = concat(attended, context_proj, behavior_embed)  -> 576-d

The backward pass propagates gradients to W_Q, W_K, W_V, W_C via the
AdamOptimizer from nn.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from local_ide_agent.rl.nn import AdamOptimizer, load_weights, save_weights, softmax


class LearnedCrossAttentionFusion:
    """
    Learned cross-attention fusion for the three state streams.

    Parameters
    ----------
    d_code:     dimensionality of the code embedding     (384)
    d_behavior: dimensionality of the behavior embedding (128)
    d_context:  dimensionality of the context embedding  (64)
    d_k:        key/query projection size                 (64)
    d_v:        value projection size                     (256)

    Output size = d_v + d_v + d_behavior = 256 + 256 + 128 = 640
    We then project down to 576-d to match the expected state vector.
    """

    def __init__(
        self,
        d_code: int = 384,
        d_behavior: int = 128,
        d_context: int = 64,
        d_k: int = 64,
        d_v: int = 256,
        d_out: int = 576,
        optimizer: AdamOptimizer | None = None,
    ) -> None:
        self.d_code = d_code
        self.d_behavior = d_behavior
        self.d_context = d_context
        self.d_k = d_k
        self.d_v = d_v
        self.d_out = d_out
        self.optimizer = optimizer or AdamOptimizer(lr=3e-4)

        scale_q = math.sqrt(2.0 / d_behavior)
        scale_k = math.sqrt(2.0 / d_code)
        scale_v = math.sqrt(2.0 / d_code)
        scale_c = math.sqrt(2.0 / d_context)
        scale_o = math.sqrt(2.0 / (d_v + d_v + d_behavior))

        # Learnable projection matrices
        self.W_Q: np.ndarray = np.random.randn(d_behavior, d_k).astype(np.float64) * scale_q
        self.W_K: np.ndarray = np.random.randn(d_code, d_k).astype(np.float64) * scale_k
        self.W_V: np.ndarray = np.random.randn(d_code, d_v).astype(np.float64) * scale_v
        self.W_C: np.ndarray = np.random.randn(d_context, d_v).astype(np.float64) * scale_c
        # Final linear to project concatenated output to d_out
        self.W_out: np.ndarray = np.random.randn(d_v + d_v + d_behavior, d_out).astype(np.float64) * scale_o
        self.b_out: np.ndarray = np.zeros(d_out, dtype=np.float64)

        # Forward-pass cache for backward
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def fuse(
        self,
        code_embed: list[float] | np.ndarray,
        behavior_embed: list[float] | np.ndarray,
        context_embed: list[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Compute attended fusion of the three embedding streams.
        Returns a 1-D array of shape (d_out,).
        """
        c = np.asarray(code_embed, dtype=np.float64)[: self.d_code]
        b = np.asarray(behavior_embed, dtype=np.float64)[: self.d_behavior]
        x = np.asarray(context_embed, dtype=np.float64)[: self.d_context]

        # Pad if shorter than expected (graceful degradation)
        c = self._pad(c, self.d_code)
        b = self._pad(b, self.d_behavior)
        x = self._pad(x, self.d_context)

        Q = b @ self.W_Q             # (d_k,)
        K = c @ self.W_K             # (d_k,)
        V = c @ self.W_V             # (d_v,)
        C_proj = x @ self.W_C       # (d_v,)

        # Scaled dot-product attention (single query/key -> scalar weight)
        raw_score = float(Q @ K) / math.sqrt(self.d_k)
        # For a single query/key pair softmax is trivially 1; use sigmoid
        attn_weight = 1.0 / (1.0 + math.exp(-raw_score))  # sigmoid
        attended = attn_weight * V   # (d_v,)

        concat = np.concatenate([attended, C_proj, b])   # (d_v + d_v + d_behavior,)
        fused = concat @ self.W_out + self.b_out          # (d_out,)

        # Cache for backward
        self._cache = {
            "c": c, "b": b, "x": x,
            "Q": Q, "K": K, "V": V, "C_proj": C_proj,
            "raw_score": raw_score, "attn_weight": attn_weight,
            "attended": attended, "concat": concat,
        }
        return fused

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, grad_fused: np.ndarray) -> None:
        """
        Back-propagate grad_fused through the fusion layer and update all
        projection matrices using the AdamOptimizer.
        """
        cache = self._cache
        if not cache:
            return

        c = cache["c"]
        b = cache["b"]
        x = cache["x"]
        Q = cache["Q"]
        K = cache["K"]
        V = cache["V"]
        attn_weight = cache["attn_weight"]
        attended = cache["attended"]
        C_proj = cache["C_proj"]
        concat = cache["concat"]
        raw_score = cache["raw_score"]

        # ---- grad through W_out ----
        grad_W_out = np.outer(concat, grad_fused)               # (d_concat, d_out)
        grad_b_out = grad_fused.copy()
        grad_concat = self.W_out @ grad_fused                   # (d_concat,)

        # Split grad_concat
        d_v = self.d_v
        d_beh = self.d_behavior
        grad_attended = grad_concat[:d_v]
        grad_C_proj = grad_concat[d_v: d_v + d_v]
        grad_b_residual = grad_concat[d_v + d_v:]

        # ---- grad through attended = attn_weight * V ----
        grad_attn_weight = float(V @ grad_attended)
        grad_V = attn_weight * grad_attended                    # (d_v,)

        # ---- grad through sigmoid(raw_score) ----
        grad_raw_score = grad_attn_weight * attn_weight * (1.0 - attn_weight)

        # ---- grad through Q @ K / sqrt(d_k) ----
        inv_sqrt_dk = 1.0 / math.sqrt(self.d_k)
        grad_Q = grad_raw_score * inv_sqrt_dk * K               # (d_k,)
        grad_K = grad_raw_score * inv_sqrt_dk * Q               # (d_k,)

        # ---- grad through linear projections ----
        grad_W_Q = np.outer(b, grad_Q)                          # (d_behavior, d_k)
        grad_W_K = np.outer(c, grad_K)                          # (d_code, d_k)
        grad_W_V = np.outer(c, grad_V)                          # (d_code, d_v)
        grad_W_C = np.outer(x, grad_C_proj)                    # (d_context, d_v)

        # ---- Adam updates ----
        self.W_Q = self.optimizer.update(self.W_Q, grad_W_Q)
        self.W_K = self.optimizer.update(self.W_K, grad_W_K)
        self.W_V = self.optimizer.update(self.W_V, grad_W_V)
        self.W_C = self.optimizer.update(self.W_C, grad_W_C)
        self.W_out = self.optimizer.update(self.W_out, grad_W_out)
        self.b_out = self.optimizer.update(self.b_out, grad_b_out)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, np.ndarray]:
        return {
            "attn_W_Q": self.W_Q,
            "attn_W_K": self.W_K,
            "attn_W_V": self.W_V,
            "attn_W_C": self.W_C,
            "attn_W_out": self.W_out,
            "attn_b_out": self.b_out,
        }

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        if "attn_W_Q" in weights:
            self.W_Q = weights["attn_W_Q"].copy()
        if "attn_W_K" in weights:
            self.W_K = weights["attn_W_K"].copy()
        if "attn_W_V" in weights:
            self.W_V = weights["attn_W_V"].copy()
        if "attn_W_C" in weights:
            self.W_C = weights["attn_W_C"].copy()
        if "attn_W_out" in weights:
            self.W_out = weights["attn_W_out"].copy()
        if "attn_b_out" in weights:
            self.b_out = weights["attn_b_out"].copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad(arr: np.ndarray, size: int) -> np.ndarray:
        if len(arr) >= size:
            return arr[:size]
        pad = np.zeros(size, dtype=np.float64)
        pad[: len(arr)] = arr
        return pad
