# centaur/io/frame_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import base64
import re
import xml.etree.ElementTree as ET
import zlib

import numpy as np
from astropy.io import fits  # type: ignore


@dataclass
class Frame:
    source: str  # "fits" or "xisf"
    path: Path
    header_cards: List[
        Dict[str, Any]
    ]  # [{"keyword": str, "value": Any, "comment": str|None}, ...]
    core: Dict[str, Any]  # normalized by CORE_FIELDS keys
    data: Optional[np.ndarray]  # float32 or None (metadata-only)


# ----------------------------
# Normalization / coercion
# ----------------------------


def _strip_wrapping_quotes(s: str) -> str:
    out = s.strip()
    # unwrap repeatedly to handle cases like "''LIGHT''" or "'IC 405'"
    for _ in range(4):
        if len(out) >= 2 and out[0] == out[-1] and out[0] in ("'", '"'):
            out = out[1:-1].strip()
        else:
            break
    return out


def _normalize_xisf_string(v: Any) -> Any:
    """
    XISF metadata values are often strings, sometimes with literal quotes.
    Normalize:
      - empty string -> None
      - strip repeated wrapping quotes
      - return clean string
    """
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return None
    return _strip_wrapping_quotes(s)


def _coerce_value(v: Any) -> Any:
    if v is None:
        return None
    try:
        if isinstance(v, (np.generic,)):
            return v.item()
    except Exception:
        pass
    return v


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, str):
            vv = v.strip()
            if not vv:
                return None
            return float(vv)
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, str):
            vv = v.strip()
            if not vv:
                return None
            return int(float(vv))
        return int(float(v))
    except Exception:
        return None


def _normalize_keyword(k: str) -> str:
    return str(k or "").strip().upper()


def _first_present_map(kv: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        kk = _normalize_keyword(k)
        if kk in kv:
            return kv.get(kk)
    return None


# ----------------------------
# Public API
# ----------------------------


def load_pixels(path: Path) -> np.ndarray:
    """
    Load pixel data as float32 for FITS or XISF.

    Returns:
      - 2D float32 image for mono
      - for multi-channel XISF, returns a 2D float32 luminance proxy (mean over channels)
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".fit", ".fits", ".fts"):
        with fits.open(p, memmap=False) as hdul:
            arr = hdul[0].data
        if arr is None:
            raise ValueError("no_image_data")
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2:
            raise ValueError(f"unsupported_fits_data_shape:{a.shape}")
        return np.asarray(a, dtype=np.float32)

    if suffix == ".xisf":
        return _load_xisf_pixels(p)

    raise ValueError(f"Unsupported file type: {p.name}")


def load_frame(
    path: Path,
    *,
    pixels: bool,
    core_fields: Optional[List[str]] = None,
    core_keywords: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Frame:
    """
    Load FITS or XISF into a shared Frame representation.

    - If pixels=False: returns metadata-only for XISF (fast).
    - If pixels=True : FITS and XISF both return pixel arrays (float32).
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".fit", ".fits", ".fts"):
        return _load_fits(
            p, pixels=pixels, core_fields=core_fields, core_keywords=core_keywords
        )

    if suffix == ".xisf":
        return _load_xisf(
            p, pixels=pixels, core_fields=core_fields, core_keywords=core_keywords
        )

    raise ValueError(f"Unsupported file type: {p.name}")


# ----------------------------
# FITS loader
# ----------------------------


def _load_fits(
    path: Path,
    *,
    pixels: bool,
    core_fields: Optional[List[str]],
    core_keywords: Optional[Dict[str, Tuple[str, ...]]],
) -> Frame:
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        cards: List[Dict[str, Any]] = []
        for card in header.cards:
            cards.append(
                {
                    "keyword": str(card.keyword),
                    "value": _coerce_value(card.value),
                    "comment": card.comment,
                }
            )

        kv: Dict[str, Any] = {}
        for c in cards:
            k = _normalize_keyword(c.get("keyword"))
            if k and k not in kv:
                kv[k] = c.get("value")

        core: Dict[str, Any] = {}
        if core_fields and core_keywords:
            for field in core_fields:
                raw = _first_present_map(kv, core_keywords[field])
                core[field] = _coerce_value(raw)

        data = None
        if pixels:
            arr = hdul[0].data
            if arr is not None:
                data = np.asarray(arr, dtype=np.float32)

    return Frame(source="fits", path=path, header_cards=cards, core=core, data=data)


# ----------------------------
# XISF helpers
# ----------------------------


def _extract_xisf_xml(
    path: Path, *, max_scan_bytes: int = 16 * 1024 * 1024
) -> Tuple[str, int, int]:
    """
    Extract the XISF XML header and return:
      (xml_text, xml_start_offset, xml_end_offset)
    """
    with path.open("rb") as f:
        blob = f.read(max_scan_bytes)

    start = blob.find(b"<xisf")
    if start < 0:
        start = blob.find(b"<XISF")
    if start < 0:
        raise ValueError("XISF XML header not found (no <xisf> tag in scan window)")

    end = blob.find(b"</xisf>")
    if end < 0:
        end = blob.find(b"</XISF>")
    if end < 0:
        raise ValueError(
            "XISF XML header not found (no </xisf> end tag in scan window)"
        )

    end = end + len(b"</xisf>")
    xml_text = blob[start:end].decode("utf-8", errors="replace")
    return xml_text, int(start), int(end)


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if tag else ""


def _parse_xisf_metadata(xml_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = ET.fromstring(xml_text)

    cards: List[Dict[str, Any]] = []
    kv: Dict[str, Any] = {}

    def add_card(keyword: str, value: Any, comment: Optional[str] = None) -> None:
        k = _normalize_keyword(keyword)
        if not k:
            return
        v = _normalize_xisf_string(value)
        cards.append({"keyword": k, "value": v, "comment": comment})
        if k not in kv:
            kv[k] = v

    for el in root.iter():
        tag_u = _local_name(el.tag).strip().upper()

        # common patterns
        if tag_u in ("FITSKEYWORD", "KEYWORD", "CARD"):
            key = (
                el.attrib.get("name")
                or el.attrib.get("keyword")
                or el.attrib.get("id")
                or el.attrib.get("key")
            )
            val = el.attrib.get("value")
            if key:
                add_card(key, val, comment="from XISF xml")
                continue

        if tag_u in ("PROPERTY", "PROP", "METADATA", "META"):
            key = (
                el.attrib.get("id")
                or el.attrib.get("name")
                or el.attrib.get("keyword")
                or el.attrib.get("key")
            )
            val = el.attrib.get("value")
            if key and val is not None:
                add_card(key, val, comment="from XISF property")
                continue

        key_attr = (
            el.attrib.get("keyword")
            or el.attrib.get("name")
            or el.attrib.get("id")
            or el.attrib.get("key")
        )
        val_attr = el.attrib.get("value")
        if key_attr and val_attr is not None:
            add_card(key_attr, val_attr, comment=f"from XISF {tag_u}")
            continue

        # tiny element text fallback
        txt = (el.text or "").strip()
        if txt and len(txt) < 200:
            if tag_u and all(ch.isalnum() or ch in ("_", "-") for ch in tag_u):
                add_card(tag_u, txt, comment="from XISF element text")

    return cards, kv


def _xisf_dtype_from_sample_format(fmt: str) -> np.dtype:
    f = (fmt or "").strip().lower()
    if "uint8" in f:
        return np.dtype("<u1")
    if "uint16" in f:
        return np.dtype("<u2")
    if "uint32" in f:
        return np.dtype("<u4")
    if "int16" in f:
        return np.dtype("<i2")
    if "int32" in f:
        return np.dtype("<i4")
    if "float32" in f or f == "float":
        return np.dtype("<f4")
    if "float64" in f or "double" in f:
        return np.dtype("<f8")
    return np.dtype("<u2")


def _parse_geometry(geom: str) -> Tuple[int, int, int]:
    g = (geom or "").strip()
    parts = [p for p in re.split(r"[:x]", g) if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"unsupported_xisf_geometry:{geom!r}")
    w = int(float(parts[0]))
    h = int(float(parts[1]))
    c = int(float(parts[2])) if len(parts) >= 3 else 1
    return w, h, c


def _parse_location(loc: str) -> Tuple[int, int]:
    s = (loc or "").strip()
    m = re.search(
        r"(?:attachment|file)\s*[:/]{1,3}\s*(\d+)\s*:\s*(\d+)", s, flags=re.IGNORECASE
    )
    if not m:
        m2 = re.search(
            r"(?:attachment|file)\s*[:/]{1,3}\s*(\d+)", s, flags=re.IGNORECASE
        )
        if m2:
            return int(m2.group(1)), -1
        raise ValueError(f"unsupported_xisf_location:{loc!r}")
    return int(m.group(1)), int(m.group(2))


def _read_block(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(size) if size >= 0 else f.read()


def _decode_xisf_data_payload(
    payload: bytes, *, compression: Optional[str], encoding: Optional[str]
) -> bytes:
    enc = (encoding or "").strip().lower()
    comp_raw = (compression or "").strip()

    raw = payload
    if enc in ("base64", "b64"):
        raw = base64.b64decode(raw)

    if not comp_raw or comp_raw.lower() in ("none",):
        return raw

    comp = comp_raw.lower()

    if comp == "zlib":
        return zlib.decompress(raw)

    if comp in ("zlib+shuffle", "zlib-shuffle"):
        raise NotImplementedError("XISF compression 'zlib+shuffle' not supported yet")

    # XISF often uses: compression="lz4:<uncompressed_bytes>"
    # and the bytes are *not* necessarily LZ4 frame format (lz4.frame), but raw LZ4 blocks.
    if comp.startswith("lz4"):
        uncompressed_size = None
        m = re.match(r"lz4\s*:\s*(\d+)", comp)
        if m:
            try:
                uncompressed_size = int(m.group(1))
            except Exception:
                uncompressed_size = None

        try:
            import lz4.block  # type: ignore

            if uncompressed_size is None:
                raise ValueError("lz4_missing_uncompressed_size")
            return lz4.block.decompress(raw, uncompressed_size=uncompressed_size)
        except Exception as e_block:
            try:
                import lz4.frame  # type: ignore

                return lz4.frame.decompress(raw)
            except Exception as e_frame:
                raise ValueError(
                    f"lz4_decompress_failed:{type(e_block).__name__}:{e_block}::{type(e_frame).__name__}:{e_frame}"
                )

    raise NotImplementedError(f"XISF compression '{compression}' not supported")


def _load_xisf_pixels(path: Path) -> np.ndarray:
    """
    XISF pixel loading:
      - locate first <Image> element
      - geometry + sampleFormat on <Image>
      - pixels referenced by:
          A) <Image ... location="attachment:..:.." compression="..."> (your file)
          B) <Image><Data location="..." compression="..." /></Image>  (other writers)
      - supports: none, zlib, lz4 (requires lz4 package)
    """
    xml_text, _xml_start, _xml_end = _extract_xisf_xml(path)
    root = ET.fromstring(xml_text)

    image_el = None
    for el in root.iter():
        if _local_name(el.tag).lower() == "image":
            image_el = el
            break
    if image_el is None:
        raise ValueError("XISF has no <Image> element")

    geom = image_el.attrib.get("geometry") or image_el.attrib.get("Geometry") or ""
    sample_fmt = (
        image_el.attrib.get("sampleFormat") or image_el.attrib.get("SampleFormat") or ""
    )
    w, h, c = _parse_geometry(geom)
    dt = _xisf_dtype_from_sample_format(sample_fmt)

    # Locate pixel payload reference
    loc = ""
    compression = None
    encoding = None

    data_el = None
    for el in image_el.iter():
        if _local_name(el.tag).lower() == "data":
            data_el = el
            break

    if data_el is not None:
        loc = data_el.attrib.get("location") or data_el.attrib.get("Location") or ""
        compression = data_el.attrib.get("compression") or data_el.attrib.get(
            "Compression"
        )
        encoding = data_el.attrib.get("encoding") or data_el.attrib.get("Encoding")
    else:
        loc = image_el.attrib.get("location") or image_el.attrib.get("Location") or ""
        compression = image_el.attrib.get("compression") or image_el.attrib.get(
            "Compression"
        )
        encoding = image_el.attrib.get("encoding") or image_el.attrib.get("Encoding")
        if not loc:
            raise ValueError(
                "XISF <Image> missing location (no <Data> and no Image@location)"
            )

    offset, size = _parse_location(loc)
    payload = _read_block(path, offset, size)
    raw = _decode_xisf_data_payload(payload, compression=compression, encoding=encoding)

    needed = int(w * h * c * dt.itemsize)
    if len(raw) < needed:
        raise ValueError(f"xisf_pixel_payload_too_small:{len(raw)}<{needed}")

    arr = np.frombuffer(raw[:needed], dtype=dt, count=w * h * c)

    if c <= 1:
        img = arr.reshape((h, w))
        return img.astype(np.float32, copy=False)

    img3 = arr.reshape((h, w, c)).astype(np.float32, copy=False)
    return np.mean(img3, axis=2).astype(np.float32, copy=False)


def _load_xisf(
    path: Path,
    *,
    pixels: bool,
    core_fields: Optional[List[str]],
    core_keywords: Optional[Dict[str, Tuple[str, ...]]],
) -> Frame:
    xml_text, _s, _e = _extract_xisf_xml(path)
    cards, kv = _parse_xisf_metadata(xml_text)

    # Keyword aliases (operate on normalized keys; values already normalized)
    alias_map: Dict[str, List[str]] = {
        "EXPTIME": ["EXPOSURE", "EXPOSURETIME", "EXPOSURE_TIME"],
        "FILTER": ["FILTERNAME", "FILTNAME", "FILTER_ID", "FILTERID"],
        "IMAGETYP": ["FRAME", "OBSTYPE", "IMAGETYPE"],
        "INSTRUME": ["CAMERA", "CAMERANAME", "INSTRUMENT"],
        "CCD-TEMP": ["CCDTEMP", "CCD_TEMP", "SENSORTMP", "TEMPCCD"],
        "DATE-OBS": ["DATEOBS", "STARTTIME", "START_TIME"],
        "BAYERPAT": ["BAYERPATTERN", "BAYER_PATTERN"],
        "GAIN": ["CAMERAGAIN"],
        "OFFSET": ["CAMERAOFFSET", "BLACKLEVEL", "BLACKLVL"],
        "XBINNING": ["XBIN"],
        "YBINNING": ["YBIN"],
    }

    for canon, aliases in alias_map.items():
        canon_u = _normalize_keyword(canon)
        if canon_u in kv and kv[canon_u] is not None:
            continue
        for a in aliases:
            au = _normalize_keyword(a)
            if au in kv and kv[au] is not None:
                kv[canon_u] = kv[au]
                break

    core: Dict[str, Any] = {}
    if core_fields and core_keywords:
        for field in core_fields:
            raw = _first_present_map(kv, core_keywords[field])
            raw = _normalize_xisf_string(raw)
            core[field] = _coerce_value(raw)

        # Coerce numeric types after string normalization
        if isinstance(core.get("exptime"), str):
            v = _as_float(core.get("exptime"))
            if v is not None:
                core["exptime"] = v

        for k in (
            "gain",
            "offset",
            "ccd_temp",
            "set_temp",
            "focallen",
            "f_ratio",
            "aperture",
            "rotator",
            "xpixsz",
            "ypixsz",
        ):
            if isinstance(core.get(k), str):
                v = _as_float(core.get(k))
                if v is not None:
                    core[k] = v

        for k in (
            "xbinning",
            "ybinning",
            "naxis1",
            "naxis2",
            "seqnum",
            "nsubexp",
            "bitpix",
        ):
            if isinstance(core.get(k), str):
                v = _as_int(core.get(k))
                if v is not None:
                    core[k] = v

        # Ensure common string fields are de-quoted
        for k in (
            "imagetyp",
            "filter",
            "object",
            "instrume",
            "detector",
            "telescop",
            "creator",
            "origin",
            "software",
            "observer",
            "project",
            "date_obs",
            "date_end",
        ):
            if isinstance(core.get(k), str):
                core[k] = _normalize_xisf_string(core.get(k))

    data = _load_xisf_pixels(path) if pixels else None
    return Frame(source="xisf", path=path, header_cards=cards, core=core, data=data)
