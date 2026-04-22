"""Smoke-test for the RND curiosity module."""
import numpy as np
from local_ide_agent.rl.curiosity import RNDModule

rnd = RNDModule(state_dim=576, embed_dim=64, beta=0.05)
sv = np.random.randn(576).tolist()

prev_r = None
for i in range(5):
    r = rnd.intrinsic_reward(sv)
    loss = rnd.update(sv)
    print(f"step {i}: r_int={r:.4f}  pred_loss={loss:.4f}")
    prev_r = r

print(f"Final intrinsic reward (should be << step 0): {rnd.intrinsic_reward(sv):.5f}")
print("RND multi-step OK")
