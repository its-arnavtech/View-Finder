"""Collect geotagged sample images from Wikimedia Commons API.

Same endpoint the View Finder landmark_builder uses. Downloads thumbnails
around a set of well-known landmarks, embeds their known GPS into EXIF so the
hybrid predictor can verify ground truth, and records a manifest.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
import piexif
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "ViewFinder-test/0.1 (local test; contact: local)",
    "Referer": "https://commons.wikimedia.org/",
}
UPLOAD_HEADERS = {
    "User-Agent": "ViewFinder-test/0.1 (local test; contact: local)",
    "Referer": "https://commons.wikimedia.org/",
}

# (label, lat, lon, radius_m) — diverse regions = "multiple data sources"
TARGETS = [
    ("eiffel_paris", 48.8584, 2.2945, 1000),
    ("golden_gate_sf", 37.8199, -122.4783, 1500),
    ("mount_rainier", 46.8523, -121.7603, 3000),
    ("colosseum_rome", 41.8902, 12.4922, 1000),
    ("tokyo_tower", 35.6586, 139.7454, 1500),
    ("statue_liberty_nyc", 40.6892, -74.0445, 1500),
    ("sydney_opera", -33.8568, 151.2153, 1500),
    ("bigben_london", 51.5007, -0.1246, 1000),
]


def _deg_to_dms(value: float):
    sign = 1 if value >= 0 else -1
    value = abs(value)
    d = int(value)
    m_full = (value - d) * 60
    m = int(m_full)
    s = (m_full - m) * 60
    return ((d, 1), (m, 1), (int(round(s * 1e6)), 1000000)), sign


def embed_gps(image_path: Path, lat: float, lon: float):
    img = Image.open(image_path)
    if img.format != "JPEG":
        img = img.convert("RGB")
    lat_dms, lat_sign = _deg_to_dms(lat)
    lon_dms, lon_sign = _deg_to_dms(lon)
    gps_ifd = {
        piexif.GPSIFD.GPSLatitude: lat_dms,
        piexif.GPSIFD.GPSLatitudeRef: ("N" if lat_sign > 0 else "S").encode("ascii"),
        piexif.GPSIFD.GPSLongitude: lon_dms,
        piexif.GPSIFD.GPSLongitudeRef: ("E" if lon_sign > 0 else "W").encode("ascii"),
    }
    exif_dict = {"GPS": gps_ifd}
    exif_bytes = piexif.dump(exif_dict)
    out = image_path.with_suffix(".jpg")
    img.save(out, "jpeg", exif=exif_bytes, quality=92)
    if out != image_path:
        image_path.unlink(missing_ok=True)
    return out


def search_geotagged(lat: float, lon: float, radius: int, limit: int = 4):
    params = {
        "action": "query",
        "generator": "geosearch",
        "ggscoord": f"{lat}|{lon}",
        "ggsradius": radius,
        "ggsnamespace": 6,
        "ggslimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata|size",
        "iiurlwidth": 800,
        "format": "json",
        "formatversion": 2,
    }
    r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=30)
    if r.status_code == 429:
        time.sleep(5)
        r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", []) or []
    out = []
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        mime = ii.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue
        thumb = ii.get("thumburl") or ii.get("url")
        if not thumb:
            continue
        # Strip utm_* params that trigger wikimedia robot policy on upload.wikimedia.org
        thumb = thumb.split("?")[0]
        out.append({
            "title": p.get("title"),
            "pageid": p.get("pageid"),
            "thumburl": thumb,
            "mime": mime,
            "license": (ii.get("extmetadata") or {}).get("LicenseShortName", {}).get("value", "?"),
            "artist": (ii.get("extmetadata") or {}).get("Artist", {}).get("value", "?"),
        })
    return out


def main(out_dir: Path, per_target: int = 3):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label, lat, lon, radius in TARGETS:
        print(f"[{label}] searching Commons near {lat},{lon} r={radius}m", flush=True)
        try:
            hits = search_geotagged(lat, lon, radius, limit=per_target + 2)
        except Exception as e:
            print(f"  search failed: {e}", flush=True)
            time.sleep(3)
            continue
        time.sleep(2)
        kept = 0
        for h in hits:
            if kept >= per_target:
                break
            try:
                ext = ".jpg" if h["mime"] == "image/jpeg" else ".png"
                fname = f"{label}_{kept+1}{ext}"
                fpath = out_dir / fname
                rr = requests.get(h["thumburl"], headers=UPLOAD_HEADERS, timeout=60)
                if rr.status_code == 429:
                    time.sleep(15)
                    rr = requests.get(h["thumburl"], headers=UPLOAD_HEADERS, timeout=60)
                rr.raise_for_status()
                fpath.write_bytes(rr.content)
                fpath = embed_gps(fpath, lat, lon)
                manifest.append({
                    "file": str(fpath),
                    "label": label,
                    "true_lat": lat,
                    "true_lon": lon,
                    "source": "wikimedia_commons",
                    "title": h["title"],
                    "license": h["license"],
                    "thumburl": h["thumburl"],
                })
                kept += 1
                print(f"  saved {fpath.name}  ({h['license']})", flush=True)
                time.sleep(3)
            except Exception as e:
                import traceback
                print(f"  download failed for {h['title']}: {e}", flush=True)
                traceback.print_exc()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nCollected {len(manifest)} images. Manifest: {out_dir/'manifest.json'}", flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample-data/commons")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(out, n)
