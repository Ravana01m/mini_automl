"""Smoke test: the Streamlit entrypoint is importable."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def test_streamlit_module_imports() -> None:
    pytest.importorskip("streamlit")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    module = importlib.import_module("app.streamlit_app")
    assert hasattr(module, "main")
