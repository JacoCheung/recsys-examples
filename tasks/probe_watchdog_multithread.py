"""Sanity check: watchdog._dump_stacks now dumps every thread's stack."""

import io
import os
import sys
import threading
import time

os.environ["RANK"] = "3"  # simulate torchrun env

sys.path.insert(
    0, "/home/scratch.junzhang_sw/workspace/github/recsys-examples/examples"
)

from commons.utils.watchdog import StackDumpWatchdog


def main():
    barrier = threading.Event()

    def worker(name):
        # Each worker just blocks until told to exit.
        barrier.wait()

    workers = []
    for i in range(2):
        t = threading.Thread(target=worker, args=(f"engine_{i}",), daemon=True)
        t.name = f"engine_{i}"
        t.start()
        workers.append(t)

    # Give workers a moment to land in barrier.wait()
    time.sleep(0.05)

    # Build a watchdog and trigger _dump_stacks manually.
    wd = StackDumpWatchdog(timeout=1, check_interval=1, on=True)
    wd._watched_thread_id = threading.current_thread().ident

    # Capture stderr
    buf = io.StringIO()
    real_stderr = sys.stderr
    sys.stderr = buf
    try:
        wd._dump_stacks(elapsed=99)
    finally:
        sys.stderr = real_stderr

    barrier.set()
    for t in workers:
        t.join(timeout=1)

    out = buf.getvalue()
    print(out)

    assert "rank3" in out, "rank tag missing in header"
    assert f"pid={os.getpid()}" in out, "pid tag missing in header"
    assert "WATCHED: MainThread" in out, "main thread not labeled WATCHED"
    assert "thread: engine_0" in out, "engine_0 worker stack missing"
    assert "thread: engine_1" in out, "engine_1 worker stack missing"
    assert (
        out.count("--- ") >= 3
    ), f"expected >= 3 thread headers, got: {out.count('--- ')}"
    print("OK: dump includes rank/pid tag + MainThread + 2 worker threads")


if __name__ == "__main__":
    main()
