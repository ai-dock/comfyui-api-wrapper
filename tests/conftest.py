"""Test bootstrap.

Adds the repo root to sys.path so `from config import ...` and
`from workers.generation_worker import ...` resolve when pytest is
invoked from anywhere.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
