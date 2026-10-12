"""Interface tests need PyQt6. Without it (a core-only install) they are not collected."""

import importlib.util

import pytest

if importlib.util.find_spec("PyQt6") is None:
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(items):
    for item in items:
        if "tests/ui" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.gui)
