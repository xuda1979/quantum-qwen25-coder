"""Reward function for code generation tasks."""

import io
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from rewards.base_reward import BaseReward

class CodeReward(BaseReward):
    """
    Reward function for code generation tasks.
    """
    def __init__(self):
        super().__init__()
        self.quantum_keywords = ["qiskit", "QuantumCircuit", "QuantumRegister", "ClassicalRegister", "execute", "Aer"]

    def get_reward(self, code_snippets):
        """
        Returns a list of rewards for the given code snippets.
        """
        rewards = []
        for code in code_snippets:
            try:
                compile(code, "<string>", "exec")

                # Pylint score
                try:
                    code_file = io.StringIO(code)
                    reporter = TextReporter(io.StringIO())
                    results = Run([code_file], reporter=reporter, exit=False)
                    pylint_score = results.linter.stats['global_note']
                except (ValueError, TypeError, IndexError):
                    pylint_score = 0.0

                # Keyword bonus
                keyword_bonus = sum(0.5 for keyword in self.quantum_keywords if keyword in code)

                reward = (pylint_score / 2.0) + keyword_bonus
                rewards.append(reward)
            except SyntaxError:
                rewards.append(-10.0)
        return rewards
