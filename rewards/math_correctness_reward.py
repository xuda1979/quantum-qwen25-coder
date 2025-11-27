"""Reward function for math correctness."""

from rewards.base_reward import BaseReward
from verifiers.math_verifier import MathVerifier

class MathCorrectnessReward(BaseReward):
    """
    Reward function for math correctness.
    """
    def __init__(self):
        super().__init__()
        self.verifier = MathVerifier()

    def get_reward(self, expressions):
        """
        Returns a list of rewards for the given mathematical expressions.
        """
        rewards = []
        for expression in expressions:
            if self.verifier.verify(expression):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
