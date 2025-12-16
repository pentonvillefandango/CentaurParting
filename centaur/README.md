# Centaur Parting

Centaur Parting is a real-time astrophotography FITS ingestion and diagnostics pipeline.

It watches one or more folders for newly written FITS files, waits until each file is stable, then runs a deterministic, modular analysis pipeline that writes **one row per image per module** into a SQLite database.

Centaur Parting is **not** an image processor.  
It is an **observational intelligence system** focused on:

- Sky noise and background characterization
- Exposure feasibility and limits
- Calibration frame (flat) quality and grouping
- Optical / PSF diagnostics
- Future real-time decision support

The database is the primary product. A GUI will consume it later.

---

## Non-Negotiable Rules

- **Strict modularity** – one module, one worker, one schema.
- **Pipeline owns `module_runs`** – workers never write to it directly.
- **Append-only metrics** – no destructive updates.
- **Logging is mandatory** for every module run.
- **SQLite DB files are never committed**.

---

## Requirements

- Python 3.14 (compatible with 3.12)
- SQLite
- macOS (current dev platform)

Core dependencies:
- numpy
- astropy
- sqlite3

---

## Project Structure

```
centaur/
├── start_centaur.py
├── pipeline.py
├── watcher.py
├── config.py
├── logging.py
├── database.py
├── init_db.py
├── sanity_check.py
│
├── fits_header_worker.py
├── sky_basic_worker.py
├── sky_background2d_worker.py
├── exposure_advice_worker.py
│
├── flat_group_worker.py
├── flat_basic_worker.py
│
├── psf_detect_worker.py
├── psf_basic_worker.py
├── psf_grid_worker.py
├── psf_model_worker.py
│
├── schema_*.py
```

Runtime artifacts (not committed):

```
data/
├── centaurparting.db
├── sanity_report_*.csv
├── sanity_summary_*.json
├── sanity_performance_*.csv
```

---

## Pipeline Routing Rules

Always runs:
- fits_header_worker

If IMAGETYP == FLAT:
- flat_group_worker
- flat_basic_worker

If IMAGETYP == LIGHT (or other):
- sky_basic_worker
- sky_background2d_worker
- exposure_advice_worker
- psf_detect_worker
- psf_basic_worker
- psf_grid_worker
- psf_model_worker

Non-applicable modules are skipped, not failed.

---

## Configuration

Configuration lives in `centaur/config.py`.

### Watch Roots

```python
watch_roots = [
  WatchRoot(
    Path("/Users/admin/Documents/Windowsshared/Astro_Data/Rig24"),
    "Rig24"
  )
]
```

### Module Enable / Disable

```python
enabled_modules = {
  "fits_header_worker": True,
  "sky_basic_worker": True,
  "sky_background2d_worker": True,
  "exposure_advice_worker": True,
  "flat_group_worker": True,
  "flat_basic_worker": True,
  "psf_detect_worker": True,
  "psf_basic_worker": True,
  "psf_grid_worker": True,
  "psf_model_worker": True,
}
```

### Logging Verbosity

Enable per-module verbose logging via:

```python
logging = LoggingConfig(
  enabled=True,
  module_verbosity={
    "fits_header_worker": False,
    "sky_basic_worker": False,
    "psf_model_worker": False,
  }
)
```

---

## Running Centaur

### Start

```bash
python -m centaur.start_centaur
```

### Stop

Press `Ctrl+C`.

Final totals and failures are printed automatically.

---

## Sanity Check

Always run after tests or large runs.

```bash
python3 centaur/sanity_check.py --db data/centaurparting.db
```

Optional explicit outputs:

```bash
python3 centaur/sanity_check.py \
  --db data/centaurparting.db \
  --out data/sanity_report.csv \
  --summary data/sanity_summary.json \
  --perf data/sanity_performance.csv
```

Outputs:
- CSV per-image report
- JSON summary
- Performance rollups

A clean run has **zero anomalies**.

---

## Database Philosophy

- One row per image per table
- All tables keyed by image_id
- Schemas are authoritative
- Additive schema changes only

### module_runs

Performance and audit spine:
- status
- expected/read/written fields
- started_utc / ended_utc
- duration_us (source of truth)
- duration_ms (derived)

---

## Known TODO

- Finalize duration_us contract + migration strategy

---

## Status

- Core pipeline stable
- Flats validated
- PSF pipeline stable and performant
- Sanity checks clean on large datasets

Centaur Parting is intentionally explicit and conservative to survive long-term development and context loss.
