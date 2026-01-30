import sys
import threading
import time
import traceback
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


class StackDumpWatchdog:
    """Watch an iterator, automatically print stack traces if no activity for a timeout period.

    This is useful for debugging hangs in training loops or other long-running iterations.
    When no heartbeat (iteration) occurs within the timeout period, the watchdog will
    automatically dump the stack trace of the watched thread to stderr.

    Args:
        timeout: Seconds of inactivity before dumping stack trace. Default: 60.
        check_interval: How often to check for timeout (seconds). Default: 10.
        on: Enable/disable the watchdog. When False, zero overhead. Default: True.

    Examples:
        Basic usage with context manager::

            with StackDumpWatchdog(timeout=60) as watchdog:
                for batch in watchdog.watch(dataloader):
                    train_step(batch)

        Multiple loops sharing one watchdog::

            with StackDumpWatchdog(timeout=60) as watchdog:
                for batch in watchdog.watch(train_loader):
                    train_step(batch)
                for batch in watchdog.watch(val_loader):
                    val_step(batch)

        Conditional enable via environment variable::

            import os
            debug_mode = os.getenv("DEBUG_WATCHDOG", "0") == "1"

            with StackDumpWatchdog(timeout=60, on=debug_mode) as watchdog:
                for batch in watchdog.watch(dataloader):
                    train_step(batch)

        One-liner using watched_iter helper::

            for batch in watched_iter(dataloader, timeout=60):
                train_step(batch)
    """

    def __init__(
        self, timeout: float = 60, check_interval: float = 10, on: bool = True
    ):
        self.timeout = timeout
        self.check_interval = check_interval
        self.on = on  # Switch
        self.last_heartbeat = time.time()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._watched_thread_id: Optional[int] = None  # ID of the thread being watched

    def _heartbeat(self):
        self.last_heartbeat = time.time()

    def _watchdog_loop(self):
        while not self._stop:
            time.sleep(self.check_interval)
            if self._stop:
                break
            elapsed = time.time() - self.last_heartbeat
            if elapsed > self.timeout:
                self._dump_stacks(elapsed)
                self._heartbeat()

    def _dump_stacks(self, elapsed: float):
        print(f"\n{'='*60}", file=sys.stderr)
        print(
            f"⚠️  WATCHDOG: No activity for {elapsed:.0f}s, dumping stack...",
            file=sys.stderr,
        )
        print(f"{'='*60}", file=sys.stderr)

        frames = sys._current_frames()

        # Only dump the watched thread's stack
        if self._watched_thread_id and self._watched_thread_id in frames:
            stack = frames[self._watched_thread_id]
            thread_name = "Unknown"
            for t in threading.enumerate():
                if t.ident == self._watched_thread_id:
                    thread_name = t.name
                    break
            print(
                f"\n--- Watched Thread: {thread_name} (id={self._watched_thread_id}) ---",
                file=sys.stderr,
            )
            traceback.print_stack(stack, file=sys.stderr)
        else:
            print(
                f"\n⚠️  Watched thread (id={self._watched_thread_id}) not found!",
                file=sys.stderr,
            )

        print(f"{'='*60}\n", file=sys.stderr)

    def _start(self):
        if not self.on:
            return
        self._stop = False
        self._heartbeat()
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()

    def _shutdown(self):
        if not self.on:
            return
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1)

    def watch(
        self, iterable: Iterable[T], start_after_first: bool = False
    ) -> Iterator[T]:
        """Wrap the iterator, updating the heartbeat with each iteration.

        Args:
            iterable: The iterable to watch.
            start_after_first: If True, start monitoring only after the first item
                is yielded. This is useful when the iterator has initialization
                overhead (e.g., dataloader prefetching). Default: False.
        """
        if not self.on:
            # If disabled, just yield from the original iterator at zero cost
            yield from iterable
        else:
            # Record the thread ID that calls watch() (the thread to monitor)
            self._watched_thread_id = threading.current_thread().ident

            first = True
            for item in iterable:
                # First yield the item, then heartbeat after user code completes
                yield item

                if first:
                    first = False
                    if start_after_first:
                        # Start monitoring after the first iteration completes
                        self._start()

                # Heartbeat AFTER yield - this is when user code has finished
                self._heartbeat()

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._shutdown()
        return False


def watched_iter(
    iterable: Iterable[T],
    timeout: float = 60,
    check_interval: float = 10,
    on: bool = True,
) -> Iterator[T]:
    """One-liner helper to wrap an iterator with stack dump watchdog.

    This is a convenience function that creates a StackDumpWatchdog and wraps
    the iterable. The watchdog starts monitoring AFTER the first item is yielded,
    so initialization overhead (e.g., dataloader prefetching) is not counted.

    Args:
        iterable: The iterable to watch (e.g., dataloader).
        timeout: Seconds of inactivity before dumping stack trace. Default: 60.
        check_interval: How often to check for timeout (seconds). Should be less
            than timeout for accurate detection. Default: 10.
        on: Enable/disable the watchdog. When False, zero overhead. Default: True.

    Yields:
        Items from the wrapped iterable.

    Examples:
        Basic usage::

            for batch in watched_iter(dataloader, timeout=60):
                train_step(batch)

        With smaller check interval for faster detection::

            for batch in watched_iter(dataloader, timeout=5, check_interval=1):
                train_step(batch)

        Disabled (zero overhead)::

            for batch in watched_iter(dataloader, timeout=60, on=False):
                train_step(batch)
    """
    if not on:
        yield from iterable
    else:
        watchdog = StackDumpWatchdog(
            timeout=timeout, check_interval=check_interval, on=True
        )
        try:
            # Use start_after_first=True so monitoring begins after first item
            yield from watchdog.watch(iterable, start_after_first=True)
        finally:
            watchdog._shutdown()
