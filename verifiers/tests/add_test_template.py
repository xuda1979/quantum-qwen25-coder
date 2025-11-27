"""Unit test for the add function."""

import unittest

class TestAdd(unittest.TestCase):
    """
    Test suite for the add function.
    """
    def test_add(self):
        """
        Tests the add function.
        """
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(-1, -1), -2)

if __name__ == "__main__":
    unittest.main()
