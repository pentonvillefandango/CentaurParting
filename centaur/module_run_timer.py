
from __future__ import annotations

import time
from dataclasses import dataclass

@dataclass(frozen=True)
class RunTiming:
    duration_us: int
    duration_ms: int

def time_run_us() -> int:
    """
    Returns a high-precision timestamp in nanoseconds.
    """
    return time.perf_counter_ns()

def timing_from_ns(start_ns: int, end_ns: int) -> RunTiming:
    dur_ns = end_ns - start_ns
    if dur_ns < 0:
        dur_ns = 0
    duration_us = dur_ns // 1_000
    duration_ms = duration_us // 1_000
    return RunTiming(duration_us=int(duration_us), duration_ms=int(duration_ms))
