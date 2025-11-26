# Agent Instructions

This document provides instructions for AI agents working on this repository.

## General Principles

- **Code Style:** All Python code should follow the PEP 8 style guide. Use a linter to ensure compliance.
- **Testing:** All new features and bug fixes must be accompanied by tests. We use `pytest` for testing.
- **Documentation:** All new functions, classes, and modules should have comprehensive docstrings. All code should be well-commented.
- **Dependencies:** All new dependencies should be added to the `requirements.txt` file with a specific version number.

## Specific Instructions

- **Data Handling:** All data handling logic should be placed in the `tools/data_utils.py` file.
- **Evaluation:** The `evaluate.py` script is used for evaluating the model. It supports both syntax checking and functional correctness evaluation.
- **Training:** The `train_sft.py` and `train_peft.py` scripts are used for training the model. These scripts should be used for all training tasks.

## Pre-commit Checks

Before submitting any changes, you must run the following checks:

- `pytest`
- `flake8`
- `mypy`

All checks must pass before a pull request will be accepted.