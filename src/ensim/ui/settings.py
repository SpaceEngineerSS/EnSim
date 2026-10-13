"""Preferences that persist between sessions (QSettings)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QByteArray, QSettings

from ensim.utils.units import UnitSystem


class Settings:
    def __init__(self, path: Path | None = None) -> None:
        # A path gives an isolated INI store (tests, screenshot scripts)
        if path is None:
            self._store = QSettings("EnSim", "EnSim")
        else:
            self._store = QSettings(str(path), QSettings.Format.IniFormat)

    def _get(self, key: str, default, kind=str):
        value = self._store.value(key, default)
        if kind is bool:
            return value in (True, "true", "1", 1)
        try:
            return kind(value)
        except (TypeError, ValueError):
            return default

    @property
    def theme(self) -> str:
        return self._get("appearance/theme", "console")

    @theme.setter
    def theme(self, value: str) -> None:
        self._store.setValue("appearance/theme", value)

    @property
    def reduce_motion(self) -> bool:
        return self._get("appearance/reduce_motion", False, bool)

    @reduce_motion.setter
    def reduce_motion(self, value: bool) -> None:
        self._store.setValue("appearance/reduce_motion", value)

    @property
    def units(self) -> UnitSystem:
        try:
            return UnitSystem(self._get("units/system", UnitSystem.SI.value))
        except ValueError:
            return UnitSystem.SI

    @units.setter
    def units(self, value: UnitSystem) -> None:
        self._store.setValue("units/system", value.value)

    @property
    def recent_files(self) -> list[Path]:
        raw = self._store.value("files/recent", [])
        items = [raw] if isinstance(raw, str) else list(raw or [])
        return [Path(item) for item in items if item]

    def add_recent(self, path: Path) -> None:
        items = [str(path)] + [str(p) for p in self.recent_files if p != path]
        self._store.setValue("files/recent", items[:8])

    def geometry(self, key: str) -> QByteArray | None:
        value = self._store.value(f"layout/{key}")
        return value if isinstance(value, QByteArray) else None

    def set_geometry(self, key: str, value: QByteArray) -> None:
        self._store.setValue(f"layout/{key}", value)

    @property
    def first_run(self) -> bool:
        return not self._get("session/seen", False, bool)

    def mark_seen(self) -> None:
        self._store.setValue("session/seen", True)
