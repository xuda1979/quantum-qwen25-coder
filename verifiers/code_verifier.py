"""Verifies code snippets by running them against a suite of unit tests."""

import os
import subprocess
import tempfile


class CodeVerifier:
    """
    Verifies code snippets by running them against a suite of unit tests.
    """
    def __init__(self, test_file):
        self.test_file = test_file

    def verify(self, code_snippet):
        """
        Runs the unit tests against the given code snippet.
        """
        with open(self.test_file, "r") as f:
            test_code = f.read()

        full_code = code_snippet + "\n" + test_code

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(full_code)
            temp_file_name = f.name

        try:
            result = subprocess.run(
                ["python", temp_file_name],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
