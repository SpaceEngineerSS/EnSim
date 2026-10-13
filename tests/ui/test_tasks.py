"""TaskRunner on real worker threads."""

import time

from PyQt6.QtCore import QEventLoop, QTimer


def _wait(qapp, condition, timeout=10.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
    return condition()


def test_result_arrives_on_the_main_thread(qapp):
    from ensim.ui.tasks import TaskRunner

    runner = TaskRunner()
    results = []
    runner.submit("add", lambda ctx: 2 + 2, on_result=results.append)
    assert _wait(qapp, lambda: results == [4])
    assert _wait(qapp, lambda: not runner.busy)


def test_cancel_stops_a_cooperative_task(qapp):
    from ensim.ui.tasks import TaskRunner

    runner = TaskRunner()
    events = []

    def spin(ctx):
        while True:
            ctx.check()
            time.sleep(0.005)

    runner.submit(
        "spin", spin, on_result=events.append, on_cancel=lambda: events.append("cancelled")
    )
    time.sleep(0.05)
    runner.cancel("spin")
    assert _wait(qapp, lambda: events == ["cancelled"])
    assert runner.shutdown(2000)


def test_errors_become_messages(qapp):
    from ensim.ui.tasks import TaskRunner

    runner = TaskRunner()
    errors = []

    def boom(ctx):
        raise KeyError("Fuel 'X' is not known")

    runner.submit("boom", boom, on_result=lambda r: None, on_error=errors.append)
    assert _wait(qapp, lambda: errors == ["Fuel 'X' is not known"])
