from __future__ import annotations

import time
from typing import Optional

from centaur.logging import Logger


def ensure_iers_ready(logger: Logger, *, timeout_s: float = 60.0) -> None:
    """
    Preload Astropy IERS tables so coordinate transforms (AltAz/airmass/moon) do not
    trigger background downloads mid-pipeline.

    This intentionally runs BEFORE the watcher loop starts.

    Notes:
    - Uses astropy's own cache and expiry logic.
    - If offline, astropy may still continue with degraded accuracy; we keep startup going.
    """
    module = "iers_preflight"
    t0 = time.perf_counter()

    try:
        from astropy.utils import iers
        from astropy.utils.iers import conf as iers_conf

        # Allow download if needed; rely on astropy's cache otherwise.
        iers_conf.auto_download = True

        # Trigger a read/download if required.
        # (This is the same pathway your AltAz transform caused.)
        _ = iers.IERS_Auto.open()

        dt = time.perf_counter() - t0
        logger.log_module_summary(
            module,
            file="(startup)",
            expected_read=1,
            read=1,
            expected_written=0,
            written=0,
            status="OK",
            duration_s=dt,
        )
    except Exception as e:
        # Do NOT hard fail startup; just log.
        dt = time.perf_counter() - t0
        logger.log_failure(
            module,
            file="(startup)",
            action="continue",
            reason=f"iers_preflight_failed: {repr(e)}",
            duration_s=dt,
        )
