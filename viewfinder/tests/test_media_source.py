"""Tests for public social-video URL resolution and the top-level CLI."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from viewfinder import cli
from viewfinder.media_source import download_social_video, is_social_video_url


def test_social_url_allowlist() -> None:
    assert is_social_video_url("https://www.instagram.com/reel/abc/")
    assert is_social_video_url("https://vm.tiktok.com/xyz/")
    assert not is_social_video_url("https://example.com/video.mp4")
    assert not is_social_video_url("file:///private/video.mp4")


def test_downloader_returns_bounded_local_file(monkeypatch, tmp_path) -> None:
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            path = tmp_path / "TikTok_123.mp4"
            path.write_bytes(b"video")
            return {
                "id": "123",
                "extractor": "TikTok",
                "extractor_key": "TikTok",
                "webpage_url": url,
                "title": "Public clip",
                "duration": 9,
                "requested_downloads": [{"filepath": str(path)}],
            }

    monkeypatch.setitem(
        sys.modules,
        "yt_dlp",
        SimpleNamespace(YoutubeDL=FakeYoutubeDL),
    )
    resolved = download_social_video(
        "https://www.tiktok.com/@example/video/123",
        download_dir=tmp_path,
    )
    assert resolved.local_path == (tmp_path / "TikTok_123.mp4").resolve()
    assert resolved.source["authentication_used"] is False
    assert resolved.source["duration_seconds"] == 9.0


def test_top_level_cli_forwards_locate_arguments(monkeypatch) -> None:
    captured = {}

    def fake_run(arguments):
        captured["arguments"] = arguments
        return 7

    monkeypatch.setattr("viewfinder.main.run", fake_run)
    assert cli.run(["locate", "clip.mp4", "--predictor", "geoclip"]) == 7
    assert captured["arguments"] == ["clip.mp4", "--predictor", "geoclip"]
