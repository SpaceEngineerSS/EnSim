"""Session log: messages, task timings and errors, in order."""

from __future__ import annotations

import logging
import time

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout

from .base import Section, caption


class _Bridge(QObject):
    line = pyqtSignal(str)


class _Handler(logging.Handler):
    def __init__(self, bridge: _Bridge):
        super().__init__(level=logging.INFO)
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        self.bridge.line.emit(f"{record.levelname.lower():8s} {record.getMessage()}")


class LogSection(Section):
    title = "Log"
    icon_name = "log"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lines: list[str] = []
        self._bridge = _Bridge()
        self._bridge.line.connect(self.add)
        self._handler = _Handler(self._bridge)
        logging.getLogger("ensim").addHandler(self._handler)

    def build(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 12, 18, 12)
        bar = QHBoxLayout()
        bar.addWidget(caption("Everything this session has done, newest last."))
        bar.addStretch()
        save = QPushButton("Save…")
        save.clicked.connect(self._save)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        bar.addWidget(save)
        bar.addWidget(clear)
        layout.addLayout(bar)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(5000)
        self.view.setPlainText("\n".join(self._lines))
        layout.addWidget(self.view, 1)
        self._root.addLayout(layout)

    def add(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        self._lines.append(line)
        del self._lines[:-5000]
        if self.built:
            self.view.appendPlainText(line)

    def _clear(self) -> None:
        self._lines.clear()
        self.view.clear()

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save log", "ensim-log.txt", "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(self._lines) + "\n")

    def detach(self) -> None:
        logging.getLogger("ensim").removeHandler(self._handler)
