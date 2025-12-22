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


def format_duration_s_style1(duration_s: Optional[float]) -> Optional[str]:
    """
    Style 1:
      - < 1s  ->  "###ms"
      - >= 1s ->  "#.##s" (under 10s), "#.#s" (10s+)
    """
    if duration_s is None:
        return None

    if duration_s < 0:
        return f"{duration_s:.2f}s"

    if duration_s < 1.0:
        ms = int(round(duration_s * 1000.0))
        return f"{ms}ms"

    if duration_s < 10.0:
        return f"{duration_s:.2f}s"

    return f"{duration_s:.1f}s"


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
        if not self._config.enabled:
            return

        with self._lock:
            self._print_header(module, file)

            line = (
                f"  {read}/{expected_read} read | "
                f"{written}/{expected_written} written | "
                f"{status}"
            )

            dur_txt = format_duration_s_style1(duration_s)
            if dur_txt is not None:
                line += f" | {dur_txt}"

            print(line)
            print()

    def log_failure(
        self,
        module: str,
        file: str,
        *,
        action: str,
        reason: str,
        duration_s: Optional[float] = None,
    ) -> None:
        if not self._config.enabled:
            return

        with self._lock:
            self._print_header(module, file)

            line = f"  FAILED | action={action} | reason={reason}"

            dur_txt = format_duration_s_style1(duration_s)
            if dur_txt is not None:
                line += f" | {dur_txt}"

            print(line)
            print()

    def log_verbose_fields(
        self,
        module: str,
        fields: Dict[str, Any],
    ) -> None:
        if not self._config.is_verbose(module):
            return
        if not self._config.enabled:
            return

        with self._lock:
            for key, value in fields.items():
                print(f"    {key}={value}")
            print()

    def _print_verbose_fields(
        self, module: str, verbose_fields: Dict[str, Any]
    ) -> None:
        """
        Supports either:
          - flat dict: {"a":1,"b":2}
          - IO dict: {"__inputs__": {...}, "__outputs__": {...}}
        """
        if not self._config.is_verbose(module):
            return

        if "__inputs__" in verbose_fields or "__outputs__" in verbose_fields:
            inputs = verbose_fields.get("__inputs__") or {}
            outputs = verbose_fields.get("__outputs__") or {}

            if inputs:
                print("    inputs:")
                for k, v in inputs.items():
                    print(f"      {k}={v}")

            if outputs:
                print("    outputs:")
                for k, v in outputs.items():
                    print(f"      {k}={v}")

            return

        # fallback: flat
        for key, value in verbose_fields.items():
            print(f"    {key}={value}")

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
        Convenience: prints summary + optional verbose fields as one atomic block.
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

            dur_txt = format_duration_s_style1(duration_s)
            if dur_txt is not None:
                line += f" | {dur_txt}"
            print(line)

            if verbose_fields is not None and self._config.is_verbose(module):
                self._print_verbose_fields(module, verbose_fields)

            print()
