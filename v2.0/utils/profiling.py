"""Opt-in Nsight Systems helpers: NVTX ranges and a capture window.

Everything here is a no-op unless the environment asks for it, so the
instrumented training/inference scripts behave exactly as before in normal runs.

Environment variables
---------------------
S2S_NVTX=1          emit NVTX ranges (shown on the NVTX row of the nsys timeline)
S2S_PROF_START=N    call cudaProfilerStart() when step counter N begins
S2S_PROF_STOP=M     call cudaProfilerStop() when step counter M begins (M > N);
                    pair with `nsys profile --capture-range=cudaProfilerApi`
S2S_PROF_EXIT=1     exit the process right after the window closes (default 1)

The "step" is whatever the caller counts: an optimizer iteration in the
training scripts, an init date in inference_postprocess_si.py.

Wall-clock time of every step is printed when the window closes, so the same
run gives a timing baseline without opening the report.
"""

import contextlib
import functools
import os
import sys
import time

import torch

NVTX_ENABLED = os.environ.get("S2S_NVTX", "0") == "1"
_PROF_START = int(os.environ.get("S2S_PROF_START", "-1"))
_PROF_STOP = int(os.environ.get("S2S_PROF_STOP", "-1"))
_PROF_EXIT = os.environ.get("S2S_PROF_EXIT", "1") == "1"


if NVTX_ENABLED:
    def nvtx_push(name: str) -> None:
        torch.cuda.nvtx.range_push(name)

    def nvtx_pop() -> None:
        torch.cuda.nvtx.range_pop()

    @contextlib.contextmanager
    def nvtx_range(name: str):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
else:
    def nvtx_push(name: str) -> None:
        pass

    def nvtx_pop() -> None:
        pass

    _NULL = contextlib.nullcontext()

    def nvtx_range(name: str):
        return _NULL


def nvtx_wrap(name: str):
    """Decorator form of nvtx_range."""
    def deco(fn):
        if not NVTX_ENABLED:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with nvtx_range(name):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def timed_iter(iterable, name: str = "data_wait"):
    """Yield from `iterable`, wrapping each fetch in an NVTX range.

    Used around DataLoader iteration so time the main process spends blocked
    waiting for the next batch shows up as its own range.
    """
    it = iter(iterable)
    while True:
        nvtx_push(name)
        try:
            item = next(it)
        except StopIteration:
            nvtx_pop()
            return
        nvtx_pop()
        yield item


class ProfileWindow:
    """Open/close the nsys capture range at chosen step counts."""

    def __init__(self, tag: str = "", rank: int = 0):
        self.enabled = _PROF_START >= 0 and _PROF_STOP > _PROF_START
        self.tag = tag
        self.rank = rank
        self.count = -1
        self._t_last = None
        self._durations = []

    def step(self) -> None:
        """Call once at the start of every step."""
        if not self.enabled:
            return
        self.count += 1
        now = time.perf_counter()
        if self._t_last is not None and self.count > _PROF_START:
            self._durations.append(now - self._t_last)
        self._t_last = now

        if self.count == _PROF_START:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            self._t_last = time.perf_counter()
            print(f"[prof {self.tag} rank{self.rank}] capture START at step {self.count}",
                  flush=True)
        elif self.count == _PROF_STOP:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            d = self._durations
            if d:
                print(f"[prof {self.tag} rank{self.rank}] capture STOP at step {self.count}: "
                      f"{len(d)} steps, mean {sum(d) / len(d):.3f}s, "
                      f"min {min(d):.3f}s, max {max(d):.3f}s, "
                      f"per-step {[round(x, 3) for x in d]}", flush=True)
            if _PROF_EXIT:
                sys.stdout.flush()
                sys.stderr.flush()
                # Skip interpreter teardown (DataLoader workers, NCCL); nsys
                # only needs the process to end after cudaProfilerStop.
                os._exit(0)
