import os
import sys
import threading
import time
import traceback
from typing import Iterable, Iterator, Optional, TypeVar

import torch

T = TypeVar("T")


class StackDumpWatchdog:
    """Iterator watchdog that dumps Python stacks after inactivity.

    This is a diagnostic utility for long-running training loops. Set
    ``on=False`` to make the wrapper a pass-through.
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
        pid = os.getpid()
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "?"

        frames = sys._current_frames()
        thread_by_id = {t.ident: t for t in threading.enumerate()}

        # Emit the whole dump in one write to reduce multi-rank stderr
        # interleaving. Put the watched thread first for readability.
        ordered_ids = []
        if self._watched_thread_id and self._watched_thread_id in frames:
            ordered_ids.append(self._watched_thread_id)
        for tid in sorted(frames.keys()):
            if tid != self._watched_thread_id:
                ordered_ids.append(tid)

        from io import StringIO

        buf = StringIO()
        buf.write("\n" + "=" * 60 + "\n")
        buf.write(
            f"⚠️  WATCHDOG [rank{rank} pid={pid}]: No activity for "
            f"{elapsed:.0f}s, dumping stacks for ALL threads...\n"
        )
        buf.write("=" * 60 + "\n")

        for tid in ordered_ids:
            stack = frames.get(tid)
            if stack is None:
                continue
            t = thread_by_id.get(tid)
            name = t.name if t is not None else "<native-only>"
            daemon = "daemon" if (t is not None and t.daemon) else "non-daemon"
            tag = "WATCHED" if tid == self._watched_thread_id else "thread"
            buf.write(
                f"\n--- [rank{rank} pid={pid}] {tag}: {name} "
                f"(tid={tid}, {daemon}) ---\n"
            )
            traceback.print_stack(stack, file=buf)

        buf.write("=" * 60 + "\n\n")
        # Single atomic-ish write to stderr. PIPE_BUF on Linux is
        # 4096; dumps larger than that may still split, but at thread-
        # boundary granularity rather than line-by-line.
        sys.stderr.write(buf.getvalue())
        sys.stderr.flush()

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


class CudaMemoryWatchdog:
    """Optional CUDA allocator defragmentation watchdog.

    Enabled with ``CUDA_MEM_WATCHDOG=1``. ``step()`` calls
    ``torch.cuda.empty_cache()`` when fragmentation or low free memory
    crosses the configured thresholds.
    """

    def __init__(
        self,
        enabled: bool = False,
        frag_threshold: float = 0.5,
        min_free_mb: int = 2048,
    ):
        self.enabled = enabled
        self.frag_threshold = frag_threshold
        self.min_free_mb = min_free_mb
        self._defrag_count = 0

    @classmethod
    def from_env(cls) -> "CudaMemoryWatchdog":
        return cls(
            enabled=os.environ.get("CUDA_MEM_WATCHDOG", "0") == "1",
            frag_threshold=float(os.environ.get("CUDA_MEM_WATCHDOG_THRESHOLD", "0.5")),
            min_free_mb=int(os.environ.get("CUDA_MEM_WATCHDOG_MIN_FREE_MB", "2048")),
        )

    def step(self) -> None:
        if not self.enabled:
            return
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        free_mb = free // (1024 * 1024)
        frag_ratio = (reserved - alloc) / total if total > 0 else 0.0

        if frag_ratio > self.frag_threshold or free_mb < self.min_free_mb:
            torch.cuda.empty_cache()
            self._defrag_count += 1
            new_free, _ = torch.cuda.mem_get_info()
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            print(
                f"[rank{rank}] [WATCHDOG] empty_cache triggered "
                f"(frag={frag_ratio:.2%}, free={free_mb}MB -> "
                f"{new_free // 1024 // 1024}MB, "
                f"total defrag count={self._defrag_count})",
                flush=True,
            )


_cuda_mem_watchdog: Optional[CudaMemoryWatchdog] = None


def get_cuda_mem_watchdog() -> CudaMemoryWatchdog:
    """Get or create the global CudaMemoryWatchdog singleton."""
    global _cuda_mem_watchdog
    if _cuda_mem_watchdog is None:
        _cuda_mem_watchdog = CudaMemoryWatchdog.from_env()
    return _cuda_mem_watchdog


def watched_iter(
    iterable: Iterable[T],
    timeout: float = 60,
    check_interval: float = 10,
    on: bool = True,
) -> Iterator[T]:
    """Wrap an iterable with ``StackDumpWatchdog``.

    Monitoring starts after the first item is yielded so dataloader
    startup does not count as an inactivity timeout.
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
