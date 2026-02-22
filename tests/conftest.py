"""Shared pytest configuration and fixtures for the FALCON test suite."""
import sys
import os

# Ensure the project root is importable from every test file.
# Individual test files may also set this themselves for standalone use.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
