"""
Comprehensive tests for the pure-NumPy neural network primitives in rl/nn.py.

Covers:
  - Activation function correctness (relu, softmax, tanh, huber)
  - Linear layer forward/backward mathematical correctness
  - MLP forward/backward gradient flow
  - AdamOptimizer convergence
  - LayerNorm forward/backward
  - Weight persistence (save/load round-trip)
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from local_ide_agent.rl.nn import (
    AdamOptimizer,
    LayerNorm,
    Linear,
    MLP,
    huber_grad,
    huber_loss,
    load_weights,
    log_softmax,
    relu,
    relu_grad,
    save_weights,
    softmax,
    tanh_act,
    tanh_grad,
)


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class TestRelu:
    def test_positive_passthrough(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_negative_zeroed(self):
        x = np.array([-1.0, -0.5, -100.0])
        np.testing.assert_array_equal(relu(x), np.zeros(3))

    def test_mixed(self):
        x = np.array([-2.0, 0.0, 3.0])
        expected = np.array([0.0, 0.0, 3.0])
        np.testing.assert_array_equal(relu(x), expected)

    def test_gradient_positive(self):
        x = np.array([1.0, 2.0])
        np.testing.assert_array_equal(relu_grad(x), np.ones(2))

    def test_gradient_negative(self):
        x = np.array([-1.0, -2.0])
        np.testing.assert_array_equal(relu_grad(x), np.zeros(2))

    def test_gradient_at_zero(self):
        x = np.array([0.0])
        # Convention: grad at 0 is 0
        assert relu_grad(x)[0] == 0.0


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_uniform_input(self):
        x = np.array([0.0, 0.0, 0.0])
        result = softmax(x)
        np.testing.assert_allclose(result, np.ones(3) / 3.0, atol=1e-6)

    def test_large_values_no_overflow(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        assert abs(result.sum() - 1.0) < 1e-5

    def test_negative_values(self):
        x = np.array([-100.0, -200.0, -300.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        assert abs(result.sum() - 1.0) < 1e-5

    def test_monotonicity(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert result[0] < result[1] < result[2]


class TestLogSoftmax:
    def test_matches_log_of_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        result = log_softmax(x)
        expected = np.log(softmax(x) + 1e-12)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_all_negative(self):
        x = np.array([1.0, 2.0, 3.0])
        result = log_softmax(x)
        assert np.all(result <= 0.0)


class TestTanh:
    def test_range(self):
        x = np.linspace(-5, 5, 100)
        result = tanh_act(x)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_zero(self):
        assert abs(tanh_act(np.array([0.0]))[0]) < 1e-10

    def test_gradient_at_zero(self):
        # tanh'(0) = 1
        assert abs(tanh_grad(np.array([0.0]))[0] - 1.0) < 1e-10

    def test_gradient_positive(self):
        x = np.array([0.5])
        # tanh'(x) = 1 - tanh(x)^2
        expected = 1.0 - np.tanh(0.5) ** 2
        assert abs(tanh_grad(x)[0] - expected) < 1e-10


class TestHuber:
    def test_quadratic_region(self):
        pred = np.array([0.5])
        target = np.array([0.0])
        # |err| = 0.5 < delta=1.0 -> 0.5 * 0.5^2 = 0.125
        loss = huber_loss(pred, target)
        assert abs(loss[0] - 0.125) < 1e-10

    def test_linear_region(self):
        pred = np.array([3.0])
        target = np.array([0.0])
        # |err| = 3.0 > delta=1.0 -> 1.0 * (3.0 - 0.5) = 2.5
        loss = huber_loss(pred, target)
        assert abs(loss[0] - 2.5) < 1e-10

    def test_gradient_quadratic(self):
        pred = np.array([0.3])
        target = np.array([0.0])
        grad = huber_grad(pred, target)
        # In quadratic region, grad = err = 0.3
        assert abs(grad[0] - 0.3) < 1e-10

    def test_gradient_linear(self):
        pred = np.array([5.0])
        target = np.array([0.0])
        grad = huber_grad(pred, target)
        # In linear region, grad = delta * sign(err) = 1.0
        assert abs(grad[0] - 1.0) < 1e-10

    def test_gradient_negative_linear(self):
        pred = np.array([-5.0])
        target = np.array([0.0])
        grad = huber_grad(pred, target)
        assert abs(grad[0] - (-1.0)) < 1e-10

    def test_zero_loss_at_target(self):
        pred = np.array([2.0])
        target = np.array([2.0])
        assert abs(huber_loss(pred, target)[0]) < 1e-10


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class TestLinear:
    def test_forward_shape(self):
        layer = Linear(4, 3, name="test")
        x = np.random.randn(4)
        y = layer.forward(x)
        assert y.shape == (3,)

    def test_forward_computation(self):
        layer = Linear(2, 2, name="test")
        layer.W = np.eye(2)
        layer.b = np.zeros(2)
        x = np.array([3.0, 7.0])
        y = layer.forward(x)
        np.testing.assert_allclose(y, x, atol=1e-10)

    def test_backward_gradient_shape(self):
        layer = Linear(4, 3, name="test")
        x = np.random.randn(4)
        _ = layer.forward(x)
        grad_out = np.ones(3)
        grad_x, grad_W, grad_b = layer.backward(grad_out)
        assert grad_x.shape == (4,)
        assert grad_W.shape == (4, 3)
        assert grad_b.shape == (3,)

    def test_backward_numerical_gradient(self):
        """Verify analytical backward matches finite-difference numerical gradient."""
        np.random.seed(42)
        layer = Linear(3, 2, name="test")
        x = np.random.randn(3)
        
        # Forward and analytical backward
        y = layer.forward(x)
        grad_out = np.array([1.0, 0.5])
        grad_x_analytical, _, _ = layer.backward(grad_out)
        
        # Numerical gradient via finite differences
        eps = 1e-5
        grad_x_numerical = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            y_plus = x_plus @ layer.W + layer.b
            
            x_minus = x.copy()
            x_minus[i] -= eps
            y_minus = x_minus @ layer.W + layer.b
            
            grad_x_numerical[i] = np.sum(grad_out * (y_plus - y_minus) / (2 * eps))
        
        np.testing.assert_allclose(grad_x_analytical, grad_x_numerical, atol=1e-5)

    def test_apply_gradients_changes_weights(self):
        layer = Linear(2, 2, name="test")
        W_before = layer.W.copy()
        x = np.random.randn(2)
        _ = layer.forward(x)
        grad_out = np.ones(2)
        _, grad_W, grad_b = layer.backward(grad_out)
        opt = AdamOptimizer(lr=0.01)
        layer.apply_gradients(grad_W, grad_b, opt)
        assert not np.allclose(layer.W, W_before)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class TestLayerNorm:
    def test_forward_zero_mean_unit_var(self):
        ln = LayerNorm()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = ln.forward(x)
        assert abs(normed.mean()) < 1e-6
        assert abs(normed.std() - 1.0) < 0.1  # approximate due to eps

    def test_backward_shape(self):
        ln = LayerNorm()
        x = np.random.randn(8)
        _ = ln.forward(x)
        grad = ln.backward(np.ones(8))
        assert grad.shape == (8,)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_forward_shape(self):
        mlp = MLP([10, 8, 4], name="test")
        x = np.random.randn(10)
        y = mlp.forward(x)
        assert y.shape == (4,)

    def test_backward_returns_correct_count(self):
        mlp = MLP([10, 8, 4], name="test")
        x = np.random.randn(10)
        _ = mlp.forward(x)
        grads = mlp.backward(np.ones(4))
        # 2 layers -> 2 (grad_W, grad_b) tuples
        assert len(grads) == 2

    def test_backward_gradient_shapes(self):
        mlp = MLP([6, 4, 2], name="test")
        x = np.random.randn(6)
        _ = mlp.forward(x)
        grads = mlp.backward(np.ones(2))
        # Layer 0: (6, 4) weights, (4,) bias
        assert grads[0][0].shape == (6, 4)
        assert grads[0][1].shape == (4,)
        # Layer 1: (4, 2) weights, (2,) bias
        assert grads[1][0].shape == (4, 2)
        assert grads[1][1].shape == (2,)

    def test_grad_norm(self):
        mlp = MLP([4, 3], name="test")
        x = np.random.randn(4)
        _ = mlp.forward(x)
        grads = mlp.backward(np.ones(3))
        norm = mlp.grad_norm(grads)
        assert norm >= 0.0
        assert np.isfinite(norm)

    def test_apply_gradients_updates_params(self):
        mlp = MLP([4, 3, 2], name="test")
        W0_before = mlp.layers[0].W.copy()
        x = np.random.randn(4)
        _ = mlp.forward(x)
        grads = mlp.backward(np.ones(2))
        opt = AdamOptimizer(lr=0.01)
        mlp.apply_gradients(grads, opt)
        assert not np.allclose(mlp.layers[0].W, W0_before)

    def test_get_set_weights_roundtrip(self):
        mlp = MLP([4, 3, 2], name="test")
        weights = mlp.get_weights()
        mlp2 = MLP([4, 3, 2], name="test")
        mlp2.set_weights(weights)
        for key in weights:
            np.testing.assert_array_equal(mlp.get_weights()[key], mlp2.get_weights()[key])


# ---------------------------------------------------------------------------
# AdamOptimizer
# ---------------------------------------------------------------------------

class TestAdamOptimizer:
    def test_single_step_changes_param(self):
        opt = AdamOptimizer(lr=0.01)
        param = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        updated = opt.update(param, grad)
        assert not np.allclose(param, updated)

    def test_converges_toward_minimum(self):
        """Simple quadratic: f(x) = x^2, grad = 2x. Should converge toward 0."""
        opt = AdamOptimizer(lr=0.1, weight_decay=0.0)
        param = np.array([5.0])
        for _ in range(200):
            grad = 2.0 * param
            param = opt.update(param, grad)
        assert abs(param[0]) < 0.5

    def test_gradient_clipping(self):
        opt = AdamOptimizer(lr=0.01, grad_clip=1.0)
        param = np.array([1.0])
        huge_grad = np.array([1000.0])
        updated = opt.update(param, huge_grad)
        # Should not explode
        assert np.isfinite(updated[0])


# ---------------------------------------------------------------------------
# Weight persistence
# ---------------------------------------------------------------------------

class TestWeightPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        weights = {
            "layer1_W": np.random.randn(4, 3),
            "layer1_b": np.random.randn(3),
            "layer2_W": np.random.randn(3, 2),
        }
        path = tmp_path / "test_weights.npz"
        save_weights(weights, path)
        loaded = load_weights(path)
        for key in weights:
            np.testing.assert_array_equal(weights[key], loaded[key])

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.npz"
        loaded = load_weights(path)
        assert loaded == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "weights.npz"
        weights = {"x": np.array([1.0])}
        save_weights(weights, path)
        loaded = load_weights(path)
        np.testing.assert_array_equal(loaded["x"], weights["x"])
