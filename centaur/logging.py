from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import threading

def log_ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def utc_now() -> str:
    """
    Return current UTC time as ISO-8601 string.
    (Used for DB timestamps, not printed in logs.)
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
    Structured, thread-safe console logger for Centaur Parting.

    Design goals:
    - Readable with long FITS filenames
    - Atomic log blocks (no interleaving across threads)
    - Supports separate expected counts for read vs write
    - No database writes
    - No business logic
    """

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config
        self._lock = threading.Lock()


    def _print_header(self, module: str, file: Optional[str]) -> None:
        ts = log_ts()
        print(f"[{ts}] {module}")
        if file:
            print(f"  file: {file}")
    def log_module_summary(
        self,
        module: str,
        file: str,
        *,
        expected_read: int,
        read: int,
        expected_written: int,
        written: int,
        status: str,
        duration_s: Optional[float] = None,
    ) -> None:
        """
        Log the standard per-module summary block.

        expected_read / expected_written allow:
          6/6 read | 12/12 written
        """
        if not self._config.enabled:
            return

        with self._lock:
            self._print_header(module, file)

            line = (
                f"  {read}/{expected_read} read | "
                f"{written}/{expected_written} written | "
                f"{status}"
            )
            if duration_s is not None:
                line += f" | {duration_s:.2f}s"

            print(line)
            print()  # blank line between blocks for readability

    def log_failure(
        self,
        module: str,
        file: str,
        *,
        action: str,
        reason: str,
        duration_s: Optional[float] = None,
    ) -> None:
        """
        Log a module failure and the configured action taken.
        """
        if not self._config.enabled:
            return

        with self._lock:
            self._print_header(module, file)

            line = f"  FAILED | action={action} | reason={reason}"
            if duration_s is not None:
                line += f" | {duration_s:.2f}s"

            print(line)
            print()  # blank line between blocks

    def log_verbose_fields(
        self,
        module: str,
        fields: Dict[str, Any],
    ) -> None:
        """
        Log verbose per-field output for a module, if enabled.

        Best practice: call this immediately after log_module_summary for the same module
        (or use log_module_result()).
        """
        if not self._config.is_verbose(module):
            return

        if not self._config.enabled:
            return

        with self._lock:
            for key, value in fields.items():
                print(f"    {key}={value}")
            print()  # blank line after verbose dump

    def log_module_result(
        self,
        module: str,
        file: str,
        *,
        expected_read: int,
        read: int,
        expected_written: int,
        written: int,
        status: str,
        duration_s: Optional[float] = None,
        verbose_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convenience: prints summary + optional verbose fields as one atomic block,
        so output never gets visually interleaved.
        """
        if not self._config.enabled:
            return

        with self._lock:
            self._print_header(module, file)

            line = (
                f"  {read}/{expected_read} read | "
                f"{written}/{expected_written} written | "
                f"{status}"
            )
            if duration_s is not None:
                line += f" | {duration_s:.2f}s"
            print(line)

            if verbose_fields is not None and self._config.is_verbose(module):
                for key, value in verbose_fields.items():
                    print(f"    {key}={value}")

            print()  # blank line between blocks
