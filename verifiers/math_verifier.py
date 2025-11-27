"""Verifies mathematical expressions using SymPy."""

import sympy

class MathVerifier:
    """
    Verifies mathematical expressions using SymPy.
    """
    def verify(self, expression):
        """
        Evaluates the given mathematical expression.
        """
        try:
            sympy.sympify(expression)
            return True
        except (SyntaxError, TypeError, sympy.SympifyError):
            return False
