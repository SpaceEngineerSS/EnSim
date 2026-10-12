from __future__ import annotations

from typing import Any

from ensim.utils import numba_support


def test_jit_disables_disk_cache_in_frozen_application(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_jit(*args: Any, **kwargs: Any):
        captured.update(kwargs)
        return args, kwargs

    monkeypatch.setattr(numba_support.sys, "frozen", True, raising=False)
    monkeypatch.setattr(numba_support, "numba_jit", fake_jit)

    numba_support.jit(nopython=True, cache=True)

    assert captured == {"nopython": True, "cache": False}


def test_jit_preserves_cache_for_python_installation(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_jit(*args: Any, **kwargs: Any):
        captured.update(kwargs)
        return args, kwargs

    monkeypatch.delattr(numba_support.sys, "frozen", raising=False)
    monkeypatch.setattr(numba_support, "numba_jit", fake_jit)

    numba_support.jit(nopython=True, cache=True)

    assert captured == {"nopython": True, "cache": True}
