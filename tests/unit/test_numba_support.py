from __future__ import annotations

from typing import Any

from ensim.utils import numba_support


def test_jit_passes_options_through(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_jit(*args: Any, **kwargs: Any):
        captured.update(kwargs)
        return args, kwargs

    monkeypatch.setattr(numba_support, "numba_jit", fake_jit)
    numba_support.jit(nopython=True, cache=True)
    assert captured == {"nopython": True, "cache": True}


def test_identity_decorator_without_numba(monkeypatch) -> None:
    monkeypatch.setattr(numba_support, "numba_jit", None)

    def square(x):
        return x * x

    assert numba_support.jit(square) is square
    assert numba_support.jit(nopython=True, cache=True)(square) is square


def test_user_cache_dir_is_absolute() -> None:
    assert numba_support.user_cache_dir().is_absolute()
