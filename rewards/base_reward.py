"""Base class for all reward functions."""

class BaseReward:
    """
    Base class for all reward functions.
    """
    def __init__(self):
        pass

    def get_reward(self, code_snippets):
        """
        Returns a list of rewards for the given code snippets.
        """
        raise NotImplementedError
