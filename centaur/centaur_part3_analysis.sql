-- ============================================================
-- Centaur Part 3 : Analysis Query Pack
-- ============================================================
-- sqlite3 -header -column data/centaurparting.db < analysis_part3.sql | tee part3_analysis_output.txt
-- ------------------------------------------------------------
-- SECTION 0 : Sanity / Inventory
-- ------------------------------------------------------------
SELECT 'SECTION 0A : Images by type' AS section;
SELECT upper(imagetyp) AS imagetyp, COUNT(*) AS n
FROM fits_header_core
GROUP BY upper(imagetyp)
ORDER BY imagetyp;

SELECT 'SECTION 0B : Module coverage for LIGHT frames' AS section;
SELECT
  COUNT(*) AS n_lights,
  SUM(sb.image_id IS NOT NULL) AS n_sky_basic,
  SUM(b2.image_id IS NOT NULL) AS n_sky_bkg2d,
  SUM(ea.image_id IS NOT NULL) AS n_exposure_advice
FROM fits_header_core f
JOIN images i USING(image_id)
LEFT JOIN sky_basic_metrics sb USING(image_id)
LEFT JOIN sky_background2d_metrics b2 USING(image_id)
LEFT JOIN exposure_advice ea USING(image_id)
WHERE upper(f.imagetyp)='LIGHT';

-- ------------------------------------------------------------
-- SECTION 1 : Core grouping (night / target / setup)
-- ------------------------------------------------------------
SELECT 'SECTION 1 : Frame counts by setup & night' AS section;
SELECT
  date(substr(f.date_obs,1,10)) AS night,
  lower(trim(f.object)) AS target,
  lower(trim(f.telescop)) AS telescope,
  lower(trim(f.instrume)) AS camera,
  lower(trim(coalesce(f.filter,'(none)'))) AS filter,
  printf('%dx%d',coalesce(f.xbinning,1),coalesce(f.ybinning,1)) AS binning,
  round(f.exptime,2) AS exptime_s,
  COUNT(*) AS n
FROM fits_header_core f
WHERE upper(f.imagetyp)='LIGHT'
GROUP BY night,target,telescope,camera,filter,binning,exptime_s
ORDER BY night DESC,n DESC;

-- ------------------------------------------------------------
-- SECTION 2 : Sky brightness & noise (per setup)
-- ------------------------------------------------------------
SELECT 'SECTION 2 : Sky rate & noise by setup' AS section;
SELECT
  date(substr(f.date_obs,1,10)) AS night,
  lower(trim(f.object)) AS target,
  lower(trim(f.instrume)) AS camera,
  lower(trim(coalesce(f.filter,'(none)'))) AS filter,
  printf('%dx%d',coalesce(f.xbinning,1),coalesce(f.ybinning,1)) AS binning,
  round(f.exptime,2) AS exptime_s,
  COUNT(*) AS n,
  round(avg(sb.ff_median_adu_s),3) AS sky_rate_adu_s,
  round(avg(sb.ff_madstd_adu_s),3) AS sky_noise_adu_s,
  round(avg(sb.ff_madstd_adu_s)/NULLIF(avg(sb.ff_median_adu_s),0),4) AS rel_noise
FROM fits_header_core f
JOIN sky_basic_metrics sb USING(image_id)
WHERE upper(f.imagetyp)='LIGHT'
GROUP BY night,target,camera,filter,binning,exptime_s
ORDER BY night DESC,n DESC;

-- ------------------------------------------------------------
-- SECTION 3 : Gradients / background structure
-- ------------------------------------------------------------
SELECT 'SECTION 3 : Gradient metrics by setup' AS section;
SELECT
  date(substr(f.date_obs,1,10)) AS night,
  lower(trim(f.object)) AS target,
  lower(trim(f.instrume)) AS camera,
  lower(trim(coalesce(f.filter,'(none)'))) AS filter,
  printf('%dx%d',coalesce(f.xbinning,1),coalesce(f.ybinning,1)) AS binning,
  round(f.exptime,2) AS exptime_s,
  COUNT(*) AS n,
  round(avg(b2.bkg2d_range_adu),1) AS bkg_range,
  round(avg(b2.corner_delta_adu),1) AS corner_delta,
  round(avg(b2.grad_p95_adu_per_tile),2) AS grad_p95
FROM fits_header_core f
JOIN sky_background2d_metrics b2 USING(image_id)
WHERE upper(f.imagetyp)='LIGHT'
GROUP BY night,target,camera,filter,binning,exptime_s
ORDER BY night DESC,n DESC;

-- ------------------------------------------------------------
-- SECTION 4 : Exposure advice vs reality
-- ------------------------------------------------------------
SELECT 'SECTION 4 : Exposure vs recommended window' AS section;
SELECT
  date(substr(f.date_obs,1,10)) AS night,
  lower(trim(f.object)) AS target,
  lower(trim(f.instrume)) AS camera,
  lower(trim(coalesce(f.filter,'(none)'))) AS filter,
  printf('%dx%d',coalesce(f.xbinning,1),coalesce(f.ybinning,1)) AS binning,
  round(f.exptime,2) AS exptime_s,
  COUNT(*) AS n,
  round(avg(ea.recommended_min_s),1) AS rec_min_s,
  round(avg(ea.recommended_max_s),1) AS rec_max_s,
  round(avg(ea.sky_limited_min_s_k5),1) AS sky_min_k5,
  round(avg(ea.gradient_limited_max_s),1) AS grad_max,
  CASE
    WHEN avg(f.exptime) < avg(ea.recommended_min_s) THEN 'too_short'
    WHEN avg(f.exptime) > avg(ea.recommended_max_s) THEN 'too_long'
    ELSE 'in_range'
  END AS verdict
FROM fits_header_core f
JOIN exposure_advice ea USING(image_id)
WHERE upper(f.imagetyp)='LIGHT'
GROUP BY night,target,camera,filter,binning,exptime_s
ORDER BY night DESC,verdict,n DESC;

-- ------------------------------------------------------------
-- SECTION 5 : Data-driven recommended sub length (per setup)
-- ------------------------------------------------------------
SELECT 'SECTION 5 : Candidate exposure ranges by setup' AS section;
SELECT
  lower(trim(f.object)) AS target,
  lower(trim(f.instrume)) AS camera,
  lower(trim(coalesce(f.filter,'(none)'))) AS filter,
  printf('%dx%d',coalesce(f.xbinning,1),coalesce(f.ybinning,1)) AS binning,
  COUNT(*) AS n_frames,
  round(avg(ea.recommended_min_s),1) AS rec_min_s,
  round(avg(ea.recommended_max_s),1) AS rec_max_s,
  round(avg(ea.sky_limited_min_s_k5),1) AS sky_min_k5,
  round(avg(ea.gradient_limited_max_s),1) AS grad_max
FROM fits_header_core f
JOIN exposure_advice ea USING(image_id)
WHERE upper(f.imagetyp)='LIGHT'
GROUP BY target,camera,filter,binning
HAVING n_frames >= 10
ORDER BY n_frames DESC;

-- ------------------------------------------------------------
-- SECTION 6 : Worst offenders (outliers)
-- ------------------------------------------------------------
SELECT 'SECTION 6A : Brightest sky frames' AS section;
SELECT
  i.image_id,
  date(substr(f.date_obs,1,10)) AS night,
  i.file_name,
  lower(trim(f.object)) AS target,
  lower(trim(coalesce(f.filter,'(none)'))) AS filter,
  round(f.exptime,2) AS exptime_s,
  round(sb.ff_median_adu_s,3) AS sky_rate_adu_s
FROM fits_header_core f
JOIN images i USING(image_id)
JOIN sky_basic_metrics sb USING(image_id)
WHERE upper(f.imagetyp)='LIGHT'
ORDER BY sb.ff_median_adu_s DESC
LIMIT 20;

SELECT 'SECTION 6B : Worst gradients' AS section;
SELECT
  i.image_id,
  date(substr(f.date_obs,1,10)) AS night,
  i.file_name,
  lower(trim(f.object)) AS target,
  lower(trim(coalesce(f.filter,'(none)'))) AS filter,
  round(f.exptime,2) AS exptime_s,
  round(b2.grad_p95_adu_per_tile,2) AS grad_p95
FROM fits_header_core f
JOIN images i USING(image_id)
JOIN sky_background2d_metrics b2 USING(image_id)
WHERE upper(f.imagetyp)='LIGHT'
ORDER BY b2.grad_p95_adu_per_tile DESC
LIMIT 20;

-- ------------------------------------------------------------
-- SECTION 7 : Flats (capture-set level)
-- ------------------------------------------------------------
SELECT 'SECTION 7 : Flat capture set statistics' AS section;
SELECT
  cs.night,
  lower(trim(coalesce(cs.target,'(none)'))) AS target,
  lower(trim(cs.camera)) AS camera,
  lower(trim(coalesce(cs.filter,'(none)'))) AS filter,
  coalesce(cs.binning,'?') AS binning,
  round(cs.exptime,2) AS exptime_s,
  COUNT(*) AS n_frames,
  round(avg(fm.median_adu),1) AS median_mean,
  round(min(fm.median_adu),1) AS median_min,
  round(max(fm.median_adu),1) AS median_max,
  round(avg(fm.gradient_p95),1) AS gradp95_mean
FROM flat_frame_links l
JOIN flat_capture_sets cs ON cs.flat_capture_set_id=l.flat_capture_set_id
JOIN flat_metrics fm USING(image_id)
GROUP BY cs.flat_capture_set_id
ORDER BY cs.night DESC,n_frames DESC;

