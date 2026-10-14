"""Resolve local media paths and public Instagram/TikTok video URLs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


LOGGER = logging.getLogger(__name__)
SUPPORTED_SOCIAL_HOSTS = ("instagram.com", "tiktok.com")
DEFAULT_DOWNLOAD_DIR = Path("local-data/social-video-cache")
DEFAULT_MAX_DURATION_SECONDS = 600
DEFAULT_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024


class MediaSourceError(RuntimeError):
    """Raised when a social URL cannot safely be resolved to one local video."""


@dataclass(frozen=True)
class ResolvedMedia:
    local_path: Path
    source: dict[str, Any]


class _YtDlpLogger:
    def debug(self, message: str) -> None:
        if not message.startswith("[debug] "):
            LOGGER.debug("%s", message)

    def info(self, message: str) -> None:
        LOGGER.info("%s", message)

    def warning(self, message: str) -> None:
        LOGGER.warning("%s", message)

    def error(self, message: str) -> None:
        LOGGER.error("%s", message)


def is_social_video_url(value: str) -> bool:
    """Return whether *value* is an allowed public Instagram/TikTok URL."""
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.rstrip(".").lower()
    return any(
        hostname == allowed or hostname.endswith(f".{allowed}")
        for allowed in SUPPORTED_SOCIAL_HOSTS
    )


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:100] or "social_video"


def _first_video(info: dict[str, Any]) -> dict[str, Any]:
    entries = info.get("entries")
    if entries is None:
        return info
    for entry in entries:
        if isinstance(entry, dict):
            return entry
    raise MediaSourceError("the URL did not contain a downloadable video")


def _downloaded_path(info: dict[str, Any], cache_dir: Path) -> Path:
    candidates: list[str] = []
    for item in info.get("requested_downloads") or []:
        if isinstance(item, dict):
            candidates.extend(
                str(item.get(key) or "") for key in ("filepath", "_filename")
            )
    candidates.extend(str(info.get(key) or "") for key in ("filepath", "_filename"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()

    video_id = _safe_component(str(info.get("id") or ""))
    matches = sorted(
        (
            path
            for path in cache_dir.glob(f"*_{video_id}.*")
            if path.is_file() and not path.name.endswith((".part", ".ytdl"))
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0].resolve()
    raise MediaSourceError("the downloader completed but no local video file was produced")


def download_social_video(
    url: str,
    *,
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> ResolvedMedia:
    """Download one public social video and return its local cache path.

    No cookies, account credentials, or browser sessions are used. Private,
    login-gated, deleted, DRM-protected, or otherwise unavailable posts fail
    with an actionable error.
    """
    if not is_social_video_url(url):
        raise MediaSourceError(
            "only public instagram.com and tiktok.com video URLs are supported"
        )
    return download_media(
        url,
        download_dir=download_dir,
        max_duration_seconds=max_duration_seconds,
        max_download_bytes=max_download_bytes,
    )


def download_media(
    url: str,
    *,
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> ResolvedMedia:
    """Download one public video from any host yt-dlp supports.

    Identical to :func:`download_social_video` minus the Instagram/TikTok
    allowlist. The CLI keeps the narrow wrapper because its documented contract
    is those two platforms; the Discord bot uses this one, where a user may
    paste a YouTube, Reddit or Twitter link and reasonably expect it to work.

    The same guards apply either way: no cookies or credentials, no live
    streams, a duration cap and a hard size cap.
    """
    if max_duration_seconds <= 0:
        raise MediaSourceError("maximum video duration must be greater than zero")
    if max_download_bytes <= 0:
        raise MediaSourceError("maximum download size must be greater than zero")
    try:
        import yt_dlp
    except ImportError as exc:
        raise MediaSourceError(
            "social URL support requires yt-dlp; install "
            "viewfinder/requirements-social-video.txt"
        ) from exc

    cache_dir = Path(download_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    def reject_unsuitable(info: dict[str, Any], *, incomplete: bool) -> str | None:
        if info.get("is_live") or info.get("live_status") in {
            "is_live",
            "is_upcoming",
        }:
            return "live streams are not supported"
        duration = info.get("duration")
        if duration is not None and float(duration) > max_duration_seconds:
            return (
                f"video duration {float(duration):.1f}s exceeds the "
                f"{max_duration_seconds}s safety limit"
            )
        return None

    options: dict[str, Any] = {
        "logger": _YtDlpLogger(),
        "quiet": True,
        "no_warnings": False,
        "noplaylist": True,
        "playlist_items": "1",
        "restrictfilenames": True,
        "overwrites": False,
        "continuedl": True,
        "max_filesize": int(max_download_bytes),
        "match_filter": reject_unsuitable,
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(cache_dir / "%(extractor)s_%(id)s.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}
        ],
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            extracted = downloader.extract_info(url, download=True)
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise MediaSourceError(
            "could not download this public post. The platform may require login, "
            f"the post may be private, or its extractor may need updating: {detail}"
        ) from exc
    if not isinstance(extracted, dict):
        raise MediaSourceError("the URL did not return video metadata")

    info = _first_video(extracted)
    local_path = _downloaded_path(info, cache_dir)
    try:
        local_path.relative_to(cache_dir)
    except ValueError as exc:
        raise MediaSourceError("downloader produced a file outside the cache directory") from exc
    size = local_path.stat().st_size
    if size <= 0 or size > max_download_bytes:
        raise MediaSourceError(
            f"downloaded video size {size} bytes violates the configured limit"
        )

    source = {
        "kind": "social_video_url",
        "original_url": url,
        "webpage_url": str(info.get("webpage_url") or url),
        "platform": str(info.get("extractor_key") or info.get("extractor") or ""),
        "media_id": str(info.get("id") or ""),
        "title": str(info.get("title") or ""),
        "uploader": str(info.get("uploader") or ""),
        "duration_seconds": (
            round(float(info["duration"]), 3) if info.get("duration") is not None else None
        ),
        "downloaded_bytes": size,
        "cached_path": str(local_path),
        "authentication_used": False,
    }
    return ResolvedMedia(local_path=local_path, source=source)
