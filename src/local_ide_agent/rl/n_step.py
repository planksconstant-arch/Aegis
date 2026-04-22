"""
N-Step Return computation for faster credit assignment.

Problem with 1-step TD
----------------------
  target = r_t + gamma * V(s_{t+1})

In a 10-step episode, reward from the final step (e.g. task solved) takes
10 individual TD updates to propagate back to step 1. This makes early
training extremely slow.

N-step solution
---------------
  target = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... + gamma^{n-1}*r_{t+n-1}
           + gamma^n * V(s_{t+n})

With n=5, attribution travels 5 hops in one update — equivalent to 5x
faster credit propagation for the first half of each episode.

Usage
-----
  nsr = NStepReturnBuffer(n=5, gamma=0.99)
  for each step in episode:
      nsr.add_step(state_vec, action_idx, reward, next_state_vec, done)
  transitions = nsr.flush()   # returns ReplayTransition list with n-step targets

  Then add these to PrioritizedReplayBuffer as normal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from local_ide_agent.rl.replay import ReplayTransition


@dataclass
class StepRecord:
    state_vector: list[float]
    action_index: int
    reward: float
    next_state_vector: list[float]
    done: bool
    context: dict = field(default_factory=dict)
    td_error: float = 1.0


class NStepReturnBuffer:
    """
    Accumulates raw environment steps and produces N-step return transitions.

    The buffer holds the last `n` steps. When full (or episode ends), it
    emits a ReplayTransition whose reward is the discounted n-step sum and
    whose next_state_vector is the state n steps ahead (or the terminal state).

    Parameters
    ----------
    n:     number of steps to look ahead
    gamma: per-step discount factor
    """

    def __init__(self, n: int = 5, gamma: float = 0.99) -> None:
        self.n = n
        self.gamma = gamma
        self._buf: deque[StepRecord] = deque(maxlen=n)
        self._pending: list[ReplayTransition] = []

    def add_step(
        self,
        state_vector: list[float],
        action_index: int,
        reward: float,
        next_state_vector: list[float],
        done: bool,
        context: dict | None = None,
        td_error: float = 1.0,
    ) -> None:
        """Add a raw (s, a, r, s', done) step to the buffer."""
        self._buf.append(
            StepRecord(
                state_vector=state_vector,
                action_index=action_index,
                reward=reward,
                next_state_vector=next_state_vector,
                done=done,
                context=context or {},
                td_error=td_error,
            )
        )

        if len(self._buf) == self.n or done:
            self._emit()

        if done:
            # Drain any remaining steps in the buffer
            while len(self._buf) > 0:
                self._emit()

    def _emit(self) -> None:
        """Compute the n-step return for the oldest step in the buffer."""
        if not self._buf:
            return

        steps = list(self._buf)
        head = steps[0]

        # Discounted sum of rewards
        n_step_reward = 0.0
        for k, step in enumerate(steps):
            n_step_reward += (self.gamma ** k) * step.reward
            if step.done:
                # Episode terminated — use 0 as next-state value
                bootstrapped_next = step.next_state_vector
                is_done = True
                self._buf.clear()
                break
        else:
            # Not terminated — bootstrap from the last step
            bootstrapped_next = steps[-1].next_state_vector
            is_done = steps[-1].done

        # Remove the oldest step (we've just emitted it)
        if self._buf:
            self._buf.popleft()

        self._pending.append(
            ReplayTransition(
                state_vector=head.state_vector,
                action_index=head.action_index,
                reward=n_step_reward,
                next_state_vector=bootstrapped_next,
                done=is_done,
                context={**head.context, "n_step": len(steps)},
                td_error=head.td_error,
            )
        )

    def flush(self) -> list[ReplayTransition]:
        """Return all completed n-step transitions and clear the pending list."""
        result = list(self._pending)
        self._pending.clear()
        return result

    def reset(self) -> None:
        """Call at the start of each new episode."""
        self._buf.clear()
        self._pending.clear()


def compute_n_step_returns(
    rewards: list[float],
    dones: list[bool],
    next_values: list[float],
    gamma: float = 0.99,
    n: int = 5,
) -> list[float]:
    """
    Batch version: given full episode rewards + bootstrap values,
    returns the n-step return for each step.

    Useful for post-episode processing when you already have the full
    trajectory collected (e.g. in the training loop after an episode ends).
    """
    T = len(rewards)
    returns = [0.0] * T
    for t in range(T):
        g = 0.0
        for k in range(n):
            if t + k >= T:
                break
            g += (gamma ** k) * rewards[t + k]
            if dones[t + k]:
                break
        else:
            # Add bootstrap from state n steps ahead (if in bounds)
            bootstrap_t = min(t + n, T - 1)
            g += (gamma ** n) * next_values[bootstrap_t] * (1.0 - float(dones[bootstrap_t]))
        returns[t] = g
    return returns
