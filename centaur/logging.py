from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


def utc_now() -> str:
    """
    Return current UTC time as ISO-8601 string.
    (Not yet used in log output, but available for future extension.)
    """
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class LoggingConfig:
    """
    Configuration for logging behavior.
    """
    enabled: bool = True
    module_verbosity: Dict[str, bool] = field(default_factory=dict)

    def is_verbose(self, module_name: str) -> bool:
        return self.module_verbosity.get(module_name, False)


class Logger:
    """
    Structured console logger for Centaur Parting.

    Design goals:
    - Human-readable, even with long FITS filenames
    - No database writes
    - No business logic
    - Deterministic, simple output
    """

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config

    def _print_header(self, module: str, file: str | None) -> None:
        print(module)
        if file:
            print(f"  file: {file}")

    def log_module_summary(
        self,
        module: str,
        file: str,
        *,
        expected: int,
        read: int,
        written: int,
        status: str,
    ) -> None:
        """
        Log the standard per-module su

