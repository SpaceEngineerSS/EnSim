"""Background work without freezing the window.

Every calculation the interface starts goes through :class:`TaskRunner`. The
function runs on Qt's thread pool and talks back only through signals, so
widgets are never touched from a worker thread. Cancelling sets a flag that
the function polls (``ctx.cancelled()``); threads are never terminated.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from ensim.core.types import SimulationCancelled

log = logging.getLogger(__name__)


class TaskContext:
    """Handed to the task function: cancellation flag and progress reporting."""

    def __init__(self, emitter: _Signals):
        self._event = threading.Event()
        self._emitter = emitter
        self._last = 0.0

    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`SimulationCancelled` if the user asked to stop."""
        if self._event.is_set():
            raise SimulationCancelled("Cancelled")

    def progress(self, fraction: float, text: str = "") -> None:
        """Report progress in [0, 1]; throttled to 20 updates per second."""
        now = time.monotonic()
        if fraction < 1.0 and now - self._last < 0.05:
            return
        self._last = now
        self._emitter.progress.emit(max(0.0, min(1.0, fraction)), text)

    def _cancel(self) -> None:
        self._event.set()


class _Signals(QObject):
    progress = pyqtSignal(float, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    done = pyqtSignal()


class Task(QRunnable):
    """One unit of background work."""

    def __init__(self, name: str, function: Callable[[TaskContext], Any]):
        super().__init__()
        self.name = name
        self.function = function
        self.signals = _Signals()
        self.context = TaskContext(self.signals)
        self.elapsed = 0.0
        self.setAutoDelete(False)

    def cancel(self) -> None:
        self.context._cancel()

    def run(self) -> None:
        start = time.perf_counter()
        try:
            result = self.function(self.context)
        except SimulationCancelled:
            self.elapsed = time.perf_counter() - start
            self.signals.cancelled.emit()
        except Exception as error:  # reported to the user, never raised into Qt
            self.elapsed = time.perf_counter() - start
            log.debug("Task %s failed:\n%s", self.name, traceback.format_exc())
            message = error.args[0] if isinstance(error, KeyError) and error.args else error
            self.signals.failed.emit(str(message) or type(error).__name__)
        else:
            self.elapsed = time.perf_counter() - start
            self.signals.succeeded.emit(result)
        finally:
            self.signals.done.emit()


class TaskRunner(QObject):
    """Starts tasks, keeps track of the running ones and cancels them on request.

    Starting a task with the name of one that is still running cancels the
    older one first, so repeated clicks never stack up work.
    """

    busy_changed = pyqtSignal(bool)
    activity = pyqtSignal(str, float, str)  # name, fraction (-1 = indeterminate), text

    def __init__(self, parent: QObject | None = None, synchronous: bool = False):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(2, QThreadPool.globalInstance().maxThreadCount() - 1))
        self._active: dict[str, Task] = {}
        self._synchronous = synchronous

    @property
    def busy(self) -> bool:
        return bool(self._active)

    def running(self, name: str) -> bool:
        return name in self._active

    def submit(
        self,
        name: str,
        function: Callable[[TaskContext], Any],
        on_result: Callable[[Any], None],
        on_error: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> Task:
        if name in self._active:
            self._active[name].cancel()
        task = Task(name, function)
        task.signals.succeeded.connect(on_result)
        if on_error:
            task.signals.failed.connect(on_error)
        if on_cancel:
            task.signals.cancelled.connect(on_cancel)
        if on_progress:
            task.signals.progress.connect(on_progress)
        task.signals.progress.connect(lambda f, text, n=name: self.activity.emit(n, f, text))
        task.signals.done.connect(lambda t=task: self._finished(t))

        was_busy = self.busy
        self._active[name] = task
        if not was_busy:
            self.busy_changed.emit(True)
        self.activity.emit(name, -1.0, "")
        if self._synchronous:
            task.run()
        else:
            self._pool.start(task)
        return task

    def cancel(self, name: str | None = None) -> None:
        targets = list(self._active.values()) if name is None else [self._active.get(name)]
        for task in targets:
            if task is not None:
                task.cancel()

    def shutdown(self, timeout_ms: int = 3000) -> bool:
        """Cancel everything and wait for the workers; True if they all stopped."""
        self.cancel()
        return self._pool.waitForDone(timeout_ms)

    def _finished(self, task: Task) -> None:
        if self._active.get(task.name) is task:
            del self._active[task.name]
        if not self.busy:
            self.busy_changed.emit(False)
