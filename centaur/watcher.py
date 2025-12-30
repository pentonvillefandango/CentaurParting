from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Iterable, Optional, Set, Tuple

from centaur.config import AppConfig, WatchRoot
from centaur.logging import Logger


@dataclass(frozen=True)
class FileReadyEvent:
    """
    Emitted when a file is considered stable and ready for processing.
    """

    file_path: Path
    watch_root_label: str
    watch_root_path: Path
    relative_path: Optional[Path]


class Watcher:
    """
    Watches one or more folders for new FITS files.

    Behaviors:
    - Ignore existing files on start (default) unless allow_backfill=True
    - File stability check: size/mtime unchanged for N seconds
    - Runs in a background thread, stoppable (GUI-friendly)
    - Emits FileReadyEvent objects into an output queue

    This module:
    - DOES NOT read FITS
    - DOES NOT write to DB
    """

    DEFAULT_FITS_EXTS = {".fits", ".fit", ".fts", ".xisf"}

    def __init__(
        self,
        config: AppConfig,
        logger: Logger,
        *,
        fits_exts: Optional[Set[str]] = None,
        out_queue: Optional[Queue[FileReadyEvent]] = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._fits_exts = {e.lower() for e in (fits_exts or self.DEFAULT_FITS_EXTS)}

        self.out_queue: Queue[FileReadyEvent] = out_queue or Queue()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Track files already seen (prevents re-processing when allow_backfill=False)
        self._seen: Set[Path] = set()

    def start(self) -> None:
        # Prevent accidental double-start
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()

        # Seed initial state (ignore existing files)
        if self._config.ignore_existing_on_start and not self._config.allow_backfill:
            self._seed_seen_from_existing()
            self._logger.log_module_summary(
                module="watcher",
                file="(startup)",
                expected_read=1,
                read=1,
                expected_written=1,
                written=1,
                status="Seeded ignore-existing set",
            )

        self._thread = threading.Thread(
            target=self._run_loop,
            name="centaur-watcher",
            daemon=True,
        )
        self._thread.start()

        self._logger.log_module_summary(
            module="watcher",
            file="(startup)",
            expected_read=1,
            read=1,
            expected_written=1,
            written=1,
            status="Started",
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        self._logger.log_module_summary(
            module="watcher",
            file="(shutdown)",
            expected_read=1,
            read=1,
            expected_written=1,
            written=1,
            status="Stopped",
        )

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _seed_seen_from_existing(self) -> None:
        for root in self._config.watch_roots:
            for p in self._iter_candidate_files(root):
                self._seen.add(p)

    def _run_loop(self) -> None:
        poll = float(self._config.stability_poll_interval_seconds)
        stability_window = int(self._config.stability_window_seconds)

        if self._config.allow_backfill:
            self._logger.log_module_summary(
                module="watcher",
                file="(startup)",
                expected_read=1,
                read=1,
                expected_written=1,
                written=1,
                status="Backfill enabled",
            )

        while not self._stop_event.is_set():
            for root in self._config.watch_roots:
                self._scan_root(root, stability_window=stability_window)

            time.sleep(poll)

    def _scan_root(self, root: WatchRoot, *, stability_window: int) -> None:
        for file_path in self._iter_candidate_files(root):
            if file_path in self._seen and not self._config.allow_backfill:
                continue

            # Mark as seen early to reduce repeated stability-check work
            self._seen.add(file_path)

            try:
                stable = self._wait_until_stable(
                    file_path, stability_window=stability_window
                )
            except FileNotFoundError:
                continue
            except PermissionError as e:
                self._logger.log_failure(
                    module="watcher",
                    file=str(file_path),
                    action="skip",
                    reason=f"permission_error:{e}",
                )
                continue
            except Exception as e:
                self._logger.log_failure(
                    module="watcher",
                    file=str(file_path),
                    action="skip",
                    reason=f"unexpected_error:{type(e).__name__}:{e}",
                )
                continue

            if not stable:
                continue

            rel = self._relative_to_root(file_path, root.root_path)

            event = FileReadyEvent(
                file_path=file_path,
                watch_root_label=root.root_label,
                watch_root_path=root.root_path,
                relative_path=rel,
            )
            self.out_queue.put(event)

            self._logger.log_module_summary(
                module="watcher",
                file=str(file_path),
                expected_read=1,
                read=1,
                expected_written=1,
                written=1,
                status="READY",
            )

    def _iter_candidate_files(self, root: WatchRoot) -> Iterable[Path]:
        root_path = root.root_path
        if not root_path.exists():
            self._logger.log_failure(
                module="watcher",
                file=str(root_path),
                action="skip",
                reason="watch_root_missing",
            )
            return []

        candidates: list[Path] = []
        try:
            for p in root_path.rglob("*"):
                if p.is_file() and p.suffix.lower() in self._fits_exts:
                    candidates.append(p)
        except PermissionError:
            return []

        return candidates

    def _relative_to_root(self, file_path: Path, root_path: Path) -> Optional[Path]:
        try:
            return file_path.relative_to(root_path)
        except Exception:
            return None

    def _stat_key(self, p: Path) -> Tuple[int, int]:
        st = p.stat()
        return (int(st.st_size), int(st.st_mtime_ns))

    def _wait_until_stable(self, p: Path, *, stability_window: int) -> bool:
        poll = float(self._config.stability_poll_interval_seconds)
        required = float(stability_window)

        start = time.monotonic()
        last_key = self._stat_key(p)
        last_change = time.monotonic()

        while not self._stop_event.is_set():
            time.sleep(poll)

            new_key = self._stat_key(p)
            if new_key != last_key:
                last_key = new_key
                last_change = time.monotonic()

            if (time.monotonic() - last_change) >= required:
                return True

            # Safety timeout (avoid hanging forever)
            if (time.monotonic() - start) > max(60.0, required * 10):
                self._logger.log_failure(
                    module="watcher",
                    file=str(p),
                    action="skip",
                    reason="stability_timeout",
                )
                return False

        return False
