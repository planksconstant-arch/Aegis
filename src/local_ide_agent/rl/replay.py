"""
Prioritized Experience Replay (PER) with SumTree.

Replaces the previous greedy top-K sample and list.pop(0) eviction
with proper stochastic priority-proportional sampling.

References
----------
Schaul et al., "Prioritized Experience Replay", ICLR 2016.

Key properties
--------------
- SumTree: O(log n) insertion and O(log n) per sample
- Priorities: p_i = (|δ_i| + ε)^α
- Importance-sampling weights: w_i = (N · P(i))^{-β} / max_w
- β anneals linearly from β_start to 1.0 over β_steps global steps
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Transition schema
# ---------------------------------------------------------------------------

@dataclass
class ReplayTransition:
    state_vector: list[float]
    action_index: int
    reward: float
    next_state_vector: list[float]
    done: bool
    context: dict[str, Any] = field(default_factory=dict)
    td_error: float = 1.0


# ---------------------------------------------------------------------------
# SumTree
# ---------------------------------------------------------------------------

class SumTree:
    """
    Binary SumTree for O(log n) priority-proportional sampling.

    Leaf nodes store individual priorities; internal nodes store sums.
    The tree is stored as a flat array of size 2*capacity.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree: list[float] = [0.0] * (2 * capacity)
        self._data: list[ReplayTransition | None] = [None] * capacity
        self._write_ptr = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, index: int, delta: float) -> None:
        parent = (index - 1) // 2
        self._tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, index: int, cumsum: float) -> int:
        """Walk the tree to find the leaf whose prefix-sum contains cumsum."""
        left = 2 * index + 1
        right = left + 1
        if left >= len(self._tree):
            return index
        if cumsum <= self._tree[left]:
            return self._retrieve(left, cumsum)
        return self._retrieve(right, cumsum - self._tree[left])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total(self) -> float:
        return self._tree[0]

    def add(self, priority: float, transition: ReplayTransition) -> None:
        leaf_index = self.capacity - 1 + self._write_ptr
        delta = priority - self._tree[leaf_index]
        self._tree[leaf_index] = priority
        self._propagate(leaf_index, delta)
        self._data[self._write_ptr] = transition
        self._write_ptr = (self._write_ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update(self, leaf_index: int, priority: float) -> None:
        """Update priority of an existing leaf (used after TD error recomputation)."""
        if leaf_index < 0 or leaf_index >= len(self._tree):
            return
        delta = priority - self._tree[leaf_index]
        self._tree[leaf_index] = priority
        self._propagate(leaf_index, delta)

    def _retrieve(self, index: int, cumsum: float) -> int:
        """Iterative walk of the tree to find the leaf containing cumsum."""
        idx = index
        while True:
            left = 2 * idx + 1
            right = left + 1
            # If left child doesn't exist, we've reached a leaf
            if left >= len(self._tree):
                return idx
            # If right child doesn't exist, go left
            if right >= len(self._tree):
                idx = left
                continue
            if cumsum <= self._tree[left] + 1e-10:
                idx = left
            else:
                cumsum -= self._tree[left]
                idx = right

    def sample_one(self, cumsum: float) -> tuple[int, float, ReplayTransition | None]:
        """
        Sample one transition with a given cumulative sum value.
        Returns (leaf_index, priority, transition).
        """
        index = self._retrieve(0, cumsum)
        # Clamp to leaf range: leaves occupy [capacity-1, 2*capacity-2]
        leaf_start = self.capacity - 1
        leaf_end = 2 * self.capacity - 1
        index = max(leaf_start, min(index, leaf_end - 1))
        priority = self._tree[index]
        data_idx = index - leaf_start
        if data_idx < 0 or data_idx >= len(self._data):
            return index, priority, None
        transition = self._data[data_idx]
        return index, priority, transition

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer backed by a SumTree.

    Parameters
    ----------
    capacity:    max number of transitions stored
    alpha:       priority exponentiation (0 = uniform, 1 = full priority)
    beta_start:  initial IS weight exponent (0 = no correction)
    beta_steps:  number of global steps over which β anneals to 1.0
    eps:         small constant to ensure non-zero priorities
    """

    def __init__(
        self,
        capacity: int = 2048,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 100_000,
        eps: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.eps = eps
        self._tree = SumTree(capacity)
        self._global_step = 0
        self._max_priority = 1.0

    # ------------------------------------------------------------------
    # β schedule
    # ------------------------------------------------------------------

    @property
    def beta(self) -> float:
        fraction = min(self._global_step / max(self.beta_steps, 1), 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, transition: ReplayTransition) -> None:
        priority = self._priority(abs(transition.td_error))
        self._tree.add(priority, transition)
        self._max_priority = max(self._max_priority, priority)

    def _priority(self, td_error: float) -> float:
        if math.isnan(td_error):
            td_error = 0.0
        if math.isinf(td_error):
            td_error = 100.0
        # Prevent massive priority spikes
        td_error = max(0.0, min(float(td_error), 1000.0))
        return (abs(td_error) + self.eps) ** self.alpha

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, batch_size: int
    ) -> tuple[list[ReplayTransition], list[float], list[int]]:
        """
        Sample `batch_size` transitions proportional to their priorities.

        Returns
        -------
        transitions:  list of ReplayTransition
        is_weights:   importance-sampling weights (normalised to max)
        leaf_indices: tree indices for priority updates
        """
        self._global_step += 1
        if len(self._tree) == 0:
            return [], [], []

        n = min(batch_size, len(self._tree))
        transitions: list[ReplayTransition] = []
        is_weights: list[float] = []
        leaf_indices: list[int] = []
        beta = self.beta
        total = self._tree.total
        n_total = len(self._tree)

        if total <= 0:
            return [], [], []

        segment = total / n
        min_prob = (self.eps ** self.alpha) / total
        max_weight = (n_total * min_prob) ** (-beta)

        for i in range(n):
            lo = segment * i
            hi = segment * (i + 1)
            cumsum = random.uniform(lo, hi)
            leaf_idx, priority, transition = self._tree.sample_one(cumsum)
            if transition is None:
                continue
            prob = priority / total
            weight = (n_total * prob) ** (-beta) / max_weight
            transitions.append(transition)
            is_weights.append(float(weight))
            leaf_indices.append(leaf_idx)

        return transitions, is_weights, leaf_indices

    # ------------------------------------------------------------------
    # Priority update
    # ------------------------------------------------------------------

    def update_priorities(self, leaf_indices: list[int], td_errors: list[float]) -> None:
        for idx, td_err in zip(leaf_indices, td_errors):
            new_priority = self._priority(abs(td_err))
            self._tree.update(idx, new_priority)
            self._max_priority = max(self._max_priority, new_priority)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tree)

    def sample_uniform(self, batch_size: int) -> list[ReplayTransition]:
        """Uniform random sample (no IS weights) for warm-start / BC."""
        transitions = [
            self._tree._data[i]
            for i in range(len(self._tree))
            if self._tree._data[i] is not None
        ]
        if not transitions:
            return []
        return random.sample(transitions, min(batch_size, len(transitions)))
