"""Test configuration that makes the package importable without installation."""

import pathlib
import sys

# Add the repository root to the import path so that ``import chess_mini_me``
# works whether or not the package has been installed.
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
