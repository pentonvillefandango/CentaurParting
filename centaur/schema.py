"""
Centaur Parting - Database Schema (v1)

This file defines the complete SQLite schema.
It contains NO logic and should not import application code.
"""

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- Improve concurrency
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

--------------------------------------------------
-- Watched roots (for path aliasing)
--------------------------------------------------
CREATE TABLE IF NOT EXISTS watch_roots (
    watch_root_id INTEGER PRIMARY KEY,
    root_path TEXT NOT NULL UNIQUE,
    root_label TEXT,
    created_utc TEXT NOT NULL
);

--------------------------------------------------
-- Images (anchor table)
--------------------------------------------------
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    relative_path TEXT,
    file_name TEXT NOT NULL,

    watch_root_id INTEGER,

    file_size_bytes INTEGER,
    file_mtime_utc TEXT,

    status TEXT NOT NULL,
    ignore_reason TEXT,

    stable_check_seconds INTEGER,
    stable_check_passed INTEGER,

    db_created_utc TEXT NOT NULL,
    db_updated_utc TEXT NOT NULL,

    FOREIGN KEY (watch_root_id)
        REFERENCES watch_roots (watch_root_id)
);

CREATE INDEX IF NOT EXISTS idx_images_status
    ON images (status);

CREATE INDEX IF NOT EXISTS idx_images_created
    ON images (db_created_utc);

--------------------------------------------------
-- Core FITS header fields (searchable)
--------------------------------------------------
CREATE TABLE IF NOT EXISTS fits_header_core (
    image_id INTEGER PRIMARY KEY,

    -- Provenance
    header_source TEXT,
    creator TEXT,
    origin TEXT,
    software TEXT,
    observer TEXT,
    project TEXT,

    -- Time & target
    date_obs TEXT,
    date_end TEXT,
    jd REAL,
    mjd_obs REAL,
    object TEXT,
    ra TEXT,
    dec TEXT,
    equinox REAL,

    -- Exposure & capture
    exptime REAL,
    exp_total REAL,
    nsubexp INTEGER,
    imagetyp TEXT,
    filter TEXT,
    seqnum INTEGER,
    gain REAL,
    offset REAL,

    -- Camera / detector
    instrume TEXT,
    detector TEXT,
    ccd_temp REAL,
    set_temp REAL,
    xbinning INTEGER,
    ybinning INTEGER,
    readmode TEXT,
    bayerpat TEXT,
    xpixsz REAL,
    ypixsz REAL,

    -- Optics / rig
    telescop TEXT,
    focallen REAL,
    f_ratio REAL,
    aperture REAL,
    rotator REAL,
    focuspos REAL,

    -- Site / environment
    sitename TEXT,
    latitude REAL,
    longitude REAL,
    elevation_m REAL,

    -- Image geometry / scaling
    naxis1 INTEGER,
    naxis2 INTEGER,
    bitpix INTEGER,
    bzero REAL,
    bscale REAL,
    datamin REAL,
    datamax REAL,

    -- WCS
    ctype1 TEXT,
    ctype2 TEXT,
    crval1 REAL,
    crval2 REAL,
    crpix1 REAL,
    crpix2 REAL,
    cdelt1 REAL,
    cdelt2 REAL,

    cd1_1 REAL,
    cd1_2 REAL,
    cd2_1 REAL,
    cd2_2 REAL,

    pc1_1 REAL,
    pc1_2 REAL,
    pc2_1 REAL,
    pc2_2 REAL,

    -- Logging / audit
    expected_fields INTEGER,
    read_fields INTEGER,
    parse_warnings TEXT,
    db_written_utc TEXT NOT NULL,

    FOREIGN KEY (image_id)
        REFERENCES images (image_id)
        ON DELETE CASCADE
);

--------------------------------------------------
-- Full FITS header (JSON dump)
--------------------------------------------------
CREATE TABLE IF NOT EXISTS fits_header_full (
    image_id INTEGER PRIMARY KEY,
    header_json TEXT NOT NULL,
    header_bytes INTEGER,
    db_written_utc TEXT NOT NULL,

    FOREIGN KEY (image_id)
        REFERENCES images (image_id)
        ON DELETE CASCADE
);

--------------------------------------------------
-- Optical setups (telescope + camera identity)
--------------------------------------------------
CREATE TABLE IF NOT EXISTS optical_setups (
    setup_id INTEGER PRIMARY KEY,
    telescop TEXT,
    instrume TEXT,
    detector TEXT,
    site_name TEXT,
    site_lat REAL,
    site_lon REAL,
    created_utc TEXT NOT NULL
);

--------------------------------------------------
-- Image to setup mapping
--------------------------------------------------
CREATE TABLE IF NOT EXISTS image_setups (
    image_id INTEGER PRIMARY KEY,
    setup_id INTEGER NOT NULL,
    method TEXT NOT NULL,
    confidence REAL,
    db_written_utc TEXT NOT NULL,

    FOREIGN KEY (image_id)
        REFERENCES images (image_id)
        ON DELETE CASCADE,

    FOREIGN KEY (setup_id)
        REFERENCES optical_setups (setup_id)
);

--------------------------------------------------
-- Module run logging
--------------------------------------------------
CREATE TABLE IF NOT EXISTS module_runs (
    run_id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL,
    module_name TEXT NOT NULL,

    expected_fields INTEGER,
    read_fields INTEGER,
    written_fields INTEGER,

    status TEXT NOT NULL,
    message TEXT,

    started_utc TEXT NOT NULL,
    ended_utc TEXT NOT NULL,
    duration_ms INTEGER,
    duration_us INTEGER,
    db_written_utc TEXT NOT NULL,

    FOREIGN KEY (image_id)
        REFERENCES images (image_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_module_runs_image
    ON module_runs (image_id);

CREATE INDEX IF NOT EXISTS idx_module_runs_module
    ON module_runs (module_name);

CREATE INDEX IF NOT EXISTS idx_module_runs_status
    ON module_runs (status);

--------------------------------------------------
-- Performance Views (GUI + sanity_check contract)
--------------------------------------------------

-- Per-module rollups by imagetyp/camera/filter
CREATE VIEW IF NOT EXISTS v_perf_module_rollup AS
SELECT
  COALESCE(UPPER(TRIM(h.imagetyp)), '') AS imagetyp,
  COALESCE(LOWER(TRIM(h.instrume)), '') AS camera,
  COALESCE(LOWER(TRIM(h.filter)), '') AS filter,
  mr.module_name AS module_name,

  COUNT(*) AS n_runs,
  AVG(CAST(mr.duration_ms AS REAL)) AS avg_duration_ms,
  MIN(mr.duration_ms) AS min_duration_ms,
  MAX(mr.duration_ms) AS max_duration_ms

FROM module_runs mr
JOIN fits_header_core h ON h.image_id = mr.image_id
WHERE mr.status = 'ok'
  AND mr.duration_ms IS NOT NULL
GROUP BY imagetyp, camera, filter, module_name;

-- Total (sum of module times) rollups by imagetyp/camera/filter
CREATE VIEW IF NOT EXISTS v_perf_total_rollup AS
WITH per_image AS (
  SELECT
    image_id,
    SUM(duration_ms) AS total_ms
  FROM module_runs
  WHERE status = 'ok'
    AND duration_ms IS NOT NULL
  GROUP BY image_id
)
SELECT
  COALESCE(UPPER(TRIM(h.imagetyp)), '') AS imagetyp,
  COALESCE(LOWER(TRIM(h.instrume)), '') AS camera,
  COALESCE(LOWER(TRIM(h.filter)), '') AS filter,

  COUNT(*) AS n_images,
  AVG(CAST(per_image.total_ms AS REAL)) AS avg_total_duration_ms,
  MIN(per_image.total_ms) AS min_total_duration_ms,
  MAX(per_image.total_ms) AS max_total_duration_ms

FROM per_image
JOIN fits_header_core h ON h.image_id = per_image.image_id
GROUP BY imagetyp, camera, filter;
"""
