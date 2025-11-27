"""Unit test for the is_palindrome function."""

import unittest

class TestIsPalindrome(unittest.TestCase):
    """
    Test suite for the is_palindrome function.
    """
    def test_is_palindrome(self):
        """
        Tests the is_palindrome function.
        """
        self.assertTrue(is_palindrome("racecar"))
        self.assertTrue(is_palindrome("level"))
        self.assertFalse(is_palindrome("hello"))
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))

if __name__ == "__main__":
    unittest.main()
