
"""
Human-friendly formatting for microsecond durations.

Style 1:
- < 1 ms      ->  "732us"
- < 1 s       ->  "14ms"
- < 10 s      ->  "1.23s"
- >= 10 s     ->  "12.3s"
"""

from __future__ import annotations


def format_duration_us(duration_us: int | None) -> str:
    if duration_us is None:
        return "n/a"

    if duration_us < 0:
        return f"{duration_us}us"

    if duration_us < 1_000:
        return f"{duration_us}us"

    if duration_us < 1_000_000:
        return f"{duration_us / 1_000:.0f}ms"

    if duration_us < 10_000_000:
        return f"{duration_us / 1_000_000:.2f}s"

    return f"{duration_us / 1_000_000:.1f}s"
