#!/usr/bin/env python3
# centaur/model_tests/run_all_model_tests.py
#
# Master test runner:
# - Creates one run directory for this invocation
# - Calls each model test script
# - Prints PASS/FAIL/ERROR per model
#
# Usage:
#   python3 centaur/model_tests/run_all_model_tests.py --db data/centaurparting.db
#   python3 centaur/model_tests/run_all_model_tests.py --db data/centaurparting.db --run-dir data/model_tests/test_results_20251220_123000

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_one(
    name: str,
    script_path: Path,
    db_path: Path,
    run_dir: Path,
) -> Tuple[str, int, str, str]:
    """
    Returns (name, exit_code, stdout, stderr).
      0 = PASS
      2 = FAIL (script-defined fail)
      1 = ERROR (script-defined error)
      other = ERROR
    """
    cmd = [
        sys.executable,
        str(script_path),
        "--db",
        str(db_path),
        "--run-dir",
        str(run_dir),
    ]

    p = subprocess.run(cmd, text=True, capture_output=True)
    return name, p.returncode, p.stdout or "", p.stderr or ""


def classify_status(code: int, stderr: str) -> str:
    """
    argparse uses exit=2 for CLI usage problems; treat those as ERROR.
    Otherwise, follow our convention:
      0 PASS
      2 FAIL
      else ERROR
    """
    low = (stderr or "").lower()
    if code == 2 and ("unrecognized arguments" in low or "usage:" in low):
        return "ERROR"
    if code == 0:
        return "PASS"
    if code == 2:
        return "FAIL"
    return "ERROR"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all Centaur model tests")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Optional: explicitly set run directory. If blank, auto: data/model_tests/test_results_<STAMP>",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    stamp = utc_stamp()
    run_dir = (
        Path(args.run_dir)
        if args.run_dir.strip()
        else (Path("data") / "model_tests" / f"test_results_{stamp}")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    tests: List[Tuple[str, str]] = [
        ("sky_basic_metrics", "centaur/model_tests/models/test_sky_basic_metrics.py"),
        (
            "sky_background2d_metrics",
            "centaur/model_tests/models/test_sky_background2d_metrics.py",
        ),
        ("saturation_metrics", "centaur/model_tests/models/test_saturation_metrics.py"),
        (
            "signal_structure_metrics",
            "centaur/model_tests/models/test_signal_structure_metrics.py",
        ),
        ("roi_signal_metrics", "centaur/model_tests/models/test_roi_signal_metrics.py"),
        (
            "nebula_mask_metrics",
            "centaur/model_tests/models/test_nebula_mask_metrics.py",
        ),
        (
            "masked_signal_metrics",
            "centaur/model_tests/models/test_masked_signal_metrics.py",
        ),
        (
            "star_headroom_metrics",
            "centaur/model_tests/models/test_star_headroom_metrics.py",
        ),
        ("psf_detect_metrics", "centaur/model_tests/models/test_psf_detect_metrics.py"),
        ("psf_basic_metrics", "centaur/model_tests/models/test_psf_basic_metrics.py"),
        ("psf_grid_metrics", "centaur/model_tests/models/test_psf_grid_metrics.py"),
        ("psf_model_metrics", "centaur/model_tests/models/test_psf_model_metrics.py"),
        ("exposure_advice", "centaur/model_tests/models/test_exposure_advice.py"),
        ("fits_header_core", "centaur/model_tests/models/test_fits_header_core.py"),
        ("fit_header_full", "centaur/model_tests/models/test_fits_header_full.py"),
        ("module_runs", "centaur/model_tests/models/test_module_runs.py"),
        ("images_table", "centaur/model_tests/models/test_images.py"),
        ("flat_frame_profiles", "centaur/model_tests/models/test_flat_profiles.py"),
        (
            "flat_frame_capture_sets ",
            "centaur/model_tests/models/test_flat_capture_sets.py",
        ),
        ("flat_frame_metrics", "centaur/model_tests/models/test_flat_metrics.py"),
        (
            "flat_frame_links",
            "centaur/model_tests/models/test_flat_frame_links.py",
        ),
        ("camera_constants", "centaur/model_tests/models/test_camera_constants.py"),
        ("watch_roots", "centaur/model_tests/models/test_watch_roots.py"),
        ("image_setups", "centaur/model_tests/models/test_image_setups.py"),
        ("optical_setups", "centaur/model_tests/models/test_optical_setups.py"),
    ]

    print(f"[run_all_model_tests] db={db_path}")
    print(f"[run_all_model_tests] run_dir={run_dir}")
    print("")

    results: List[Tuple[str, str, int]] = []

    for name, rel_path in tests:
        script_path = Path(rel_path)
        print(f"[run_all_model_tests] running {name} ...")

        n, code, out, err = run_one(name, script_path, db_path, run_dir)
        status = classify_status(code, err)
        results.append((n, status, code))

        if out.strip():
            for line in out.rstrip().splitlines():
                print(f"    {line}")
        if err.strip():
            for line in err.rstrip().splitlines():
                print(f"    [stderr] {line}")

        print(f"[run_all_model_tests] {name}: {status} (exit={code})")
        print("")

    any_fail = any(s == "FAIL" for _, s, _ in results)
    any_error = any(s == "ERROR" for _, s, _ in results)

    print("=== SUMMARY ===")
    for name, status, _code in results:
        print(f"{name:28s} {status}")

    if any_error:
        return 1
    if any_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
