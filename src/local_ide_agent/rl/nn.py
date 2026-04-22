"""
Pure-numpy neural network primitives for the RL policy.

Provides:
  - Linear layer with He init, forward, and backward passes
  - Activation functions: relu, softmax, tanh + their derivatives
  - MLP: stacked Linear layers with relu activations
  - LayerNorm: running-stat free layer normalization
  - AdamOptimizer: per-parameter Adam state with bias correction
  - Weight persistence: save / load via numpy .npz archives
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of relu w.r.t. pre-activation x."""
    return (x > 0).astype(x.dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-8)


def log_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True) + 1e-8)


def tanh_act(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def huber_loss(pred: np.ndarray, target: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Element-wise Huber loss."""
    err = pred - target
    abs_err = np.abs(err)
    quad = 0.5 * err ** 2
    lin = delta * (abs_err - 0.5 * delta)
    return np.where(abs_err <= delta, quad, lin)


def huber_grad(pred: np.ndarray, target: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Gradient of Huber loss w.r.t. pred."""
    err = pred - target
    abs_err = np.abs(err)
    return np.where(abs_err <= delta, err, delta * np.sign(err))


# ---------------------------------------------------------------------------
# AdamOptimizer
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """Per-parameter group Adam optimizer."""

    def __init__(
        self,
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self._state: dict[int, dict[str, Any]] = {}
        self.step_count = 0

    def _get_state(self, param_id: int, shape: tuple[int, ...]) -> dict[str, Any]:
        if param_id not in self._state:
            self._state[param_id] = {
                "m": np.zeros(shape, dtype=np.float64),
                "v": np.zeros(shape, dtype=np.float64),
                "t": 0,
            }
        return self._state[param_id]

    def update(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Apply one Adam step with gradient clipping; returns updated parameter array."""
        pid = id(param)
        state = self._get_state(pid, param.shape)
        state["t"] += 1
        t = state["t"]

        # L2 weight decay applied to gradient
        g = grad + self.weight_decay * param

        # Gradient clipping by norm
        g_norm = float(np.linalg.norm(g))
        if g_norm > self.grad_clip:
            g = g * (self.grad_clip / (g_norm + 1e-8))

        state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * g
        state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * g ** 2

        m_hat = state["m"] / (1.0 - self.beta1 ** t)
        v_hat = state["v"] / (1.0 - self.beta2 ** t)

        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "step_count": self.step_count,
        }


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class Linear:
    """
    Fully-connected linear layer: y = x @ W + b

    Supports forward and backward passes for manual gradient computation.
    He normal initialization for weights.
    """

    def __init__(self, in_features: int, out_features: int, name: str = "") -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.name = name or f"linear_{in_features}x{out_features}"

        # He normal init: std = sqrt(2 / fan_in)
        std = math.sqrt(2.0 / in_features)
        self.W: np.ndarray = np.random.randn(in_features, out_features).astype(np.float64) * std
        self.b: np.ndarray = np.zeros(out_features, dtype=np.float64)

        # Cache for backward
        self._last_input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., in_features) -> (..., out_features)"""
        self._last_input = x.copy()
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (grad_x, grad_W, grad_b).
        grad_out: (..., out_features)
        """
        assert self._last_input is not None, "backward called before forward"
        x = self._last_input
        # Handle batched (2-D) and single (1-D) inputs
        if x.ndim == 1:
            grad_W = np.outer(x, grad_out)
            grad_b = grad_out.copy()
            grad_x = self.W @ grad_out
        else:
            grad_W = x.T @ grad_out
            grad_b = grad_out.sum(axis=0)
            grad_x = grad_out @ self.W.T
        return grad_x, grad_W, grad_b

    def apply_gradients(self, grad_W: np.ndarray, grad_b: np.ndarray, optimizer: AdamOptimizer) -> None:
        self.W = optimizer.update(self.W, grad_W)
        self.b = optimizer.update(self.b, grad_b)

    def parameters(self) -> list[tuple[str, np.ndarray]]:
        return [(f"{self.name}.W", self.W), (f"{self.name}.b", self.b)]


# ---------------------------------------------------------------------------
# Layer normalization
# ---------------------------------------------------------------------------

class LayerNorm:
    """Simple layer normalization (no learnable scale/shift for simplicity)."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self._last_x: np.ndarray | None = None
        self._last_norm: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_x = x
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + self.eps)
        self._last_norm = normed
        return normed

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Simplified backward: treat as identity-scale LN gradient."""
        assert self._last_x is not None
        n = self._last_x.shape[-1]
        x = self._last_x
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        grad_x = (1.0 / n) * (1.0 / std) * (
            n * grad_out
            - grad_out.sum(axis=-1, keepdims=True)
            - x_norm * (grad_out * x_norm).sum(axis=-1, keepdims=True)
        )
        return grad_x


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP:
    """
    Multi-layer perceptron with relu activations between hidden layers.

    layer_sizes example: [576, 256, 128] -> two linear layers
    The final layer has NO activation (raw logits / features).
    """

    def __init__(self, layer_sizes: list[int], name: str = "mlp") -> None:
        assert len(layer_sizes) >= 2, "Need at least input and output sizes"
        self.layer_sizes = layer_sizes
        self.name = name
        self.layers: list[Linear] = []
        self.layer_norm = LayerNorm()
        for idx in range(len(layer_sizes) - 1):
            self.layers.append(
                Linear(layer_sizes[idx], layer_sizes[idx + 1], name=f"{name}_l{idx}")
            )
        # Cache activations for backward
        self._activations: list[np.ndarray] = []
        self._pre_activations: list[np.ndarray] = []

    def forward(self, x: np.ndarray | list[float]) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        # Apply layer norm to input
        h = self.layer_norm.forward(h)
        self._activations = [h]
        self._pre_activations = []
        for idx, layer in enumerate(self.layers):
            pre = layer.forward(h)
            self._pre_activations.append(pre)
            if idx < len(self.layers) - 1:
                h = relu(pre)
            else:
                h = pre  # No activation on final layer
            self._activations.append(h)
        return h

    def backward(self, grad_out: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Backpropagate grad_out through all layers.
        Returns list of (grad_W, grad_b) per layer (in forward order).
        """
        grads: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.layers)  # type: ignore[list-item]
        g = grad_out
        for idx in reversed(range(len(self.layers))):
            if idx < len(self.layers) - 1:
                # Apply relu gradient
                g = g * relu_grad(self._pre_activations[idx])
            grad_x, grad_W, grad_b = self.layers[idx].backward(g)
            grads[idx] = (grad_W, grad_b)
            g = grad_x
        # Propagate through layer norm
        _ = self.layer_norm.backward(g)
        return grads

    def apply_gradients(
        self, grads: list[tuple[np.ndarray, np.ndarray]], optimizer: AdamOptimizer
    ) -> None:
        for layer, (grad_W, grad_b) in zip(self.layers, grads):
            layer.apply_gradients(grad_W, grad_b, optimizer)

    def grad_norm(self, grads: list[tuple[np.ndarray, np.ndarray]]) -> float:
        total_sq = sum(
            float(np.sum(gW ** 2) + np.sum(gb ** 2)) for gW, gb in grads
        )
        return math.sqrt(total_sq)

    def parameters(self) -> list[tuple[str, np.ndarray]]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def get_weights(self) -> dict[str, np.ndarray]:
        return {name: arr.copy() for name, arr in self.parameters()}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        for layer in self.layers:
            for name, arr in layer.parameters():
                if name in weights:
                    if name.endswith(".W"):
                        layer.W = weights[name].copy()
                    elif name.endswith(".b"):
                        layer.b = weights[name].copy()


# ---------------------------------------------------------------------------
# Weight persistence helpers
# ---------------------------------------------------------------------------

def save_weights(weight_dict: dict[str, np.ndarray], path: str | Path) -> None:
    """Save a flat dict of named numpy arrays to a .npz file."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(save_path), **weight_dict)


def load_weights(path: str | Path) -> dict[str, np.ndarray]:
    """Load a .npz file back into a dict of numpy arrays."""
    load_path = Path(path)
    if not load_path.exists():
        return {}
    with np.load(str(load_path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
