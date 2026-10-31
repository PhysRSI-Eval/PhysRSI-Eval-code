"""C1 harness smoke policy (NOT a learned policy): holds the current joints and gripper for H ticks. Validates bc_rollout.py end to end."""
import numpy as np


class _Hold:
    def act(self, obs):
        return dict(kind='joint', actions=np.repeat(np.asarray(obs['state'], float)[None], 16, 0))


def load(ckpt):
    return _Hold()
