"""
A bounded set of LLM calls (fixes weakness #18).

Before Day 16 every LLM call got its own daemon thread, and a call that overran its timeout was simply abandoned:
under load (or a hung Gemini) those orphan threads piled up without limit. Now:
  - at most LLM_POOL_SIZE calls run at once (default 4), each on a daemon thread;
  - if all slots are taken we don't queue - the caller gets LLMBusy at once and uses its fallback;
  - a call that overruns its timeout keeps its slot until it really ends, and the Gemini client has its own HTTP
    timeout (llm_client.HTTP_TIMEOUT_MS), so a hung call can't hold a slot forever.
Daemon threads (not a ThreadPoolExecutor): the executor's workers are joined at interpreter exit, so one hung call
would block shutdown.
"""

from __future__ import annotations

import os
import threading

POOL_SIZE = int(os.environ.get("LLM_POOL_SIZE", "4"))


class LLMTimeout(RuntimeError):
    """The LLM call did not return within the wall-clock budget."""


class LLMBusy(RuntimeError):
    """Every slot is already taken by a running LLM call."""


class LLMPool:
    def __init__(self, size: int = POOL_SIZE):
        self.size = size
        self._slots = threading.BoundedSemaphore(size)
        self._lock = threading.Lock()
        self._running = 0

    def in_flight(self) -> int:
        return self._running

    def run(self, fn, arg, timeout_s: float):
        """fn(arg) on its own daemon thread, waiting at most timeout_s. Raises LLMBusy, LLMTimeout or fn's error."""
        if not self._slots.acquire(blocking=False):
            raise LLMBusy(f"all {self.size} LLM slots busy")
        box: dict = {}
        done = threading.Event()

        def target():
            try:
                box["result"] = fn(arg)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc
            finally:
                with self._lock:
                    self._running -= 1
                self._slots.release()       # the slot frees when the call really ends, not at the timeout
                done.set()

        with self._lock:
            self._running += 1
        try:
            threading.Thread(target=target, daemon=True, name="llm-call").start()
        except BaseException:
            with self._lock:
                self._running -= 1
            self._slots.release()
            raise
        if not done.wait(max(0.0, timeout_s)):
            raise LLMTimeout(f"no response within {timeout_s:.2f}s")
        if "error" in box:
            raise box["error"]
        return box.get("result")


_default = LLMPool()
run = _default.run
in_flight = _default.in_flight
