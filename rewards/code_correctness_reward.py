"""Reward function for code correctness."""

from rewards.base_reward import BaseReward
from verifiers.code_verifier import CodeVerifier

class CodeCorrectnessReward(BaseReward):
    """
    Reward function for code correctness.
    """
    def __init__(self, test_file):
        super().__init__()
        self.verifier = CodeVerifier(test_file)

    def get_reward(self, code_snippets):
        """
        Returns a list of rewards for the given code snippets.
        """
        rewards = []
        for code in code_snippets:
            if self.verifier.verify(code):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
