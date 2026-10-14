"""Top-level subcommand CLI for View Finder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from . import __version__


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="view-finder",
        description=(
            "Local-first image and video geolocation with explicit REVIEW and "
            "ABSTAIN outcomes."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")

    locate = subcommands.add_parser(
        "locate",
        add_help=False,
        help="locate a local image/video or public Instagram/TikTok video",
    )
    locate.add_argument("arguments", nargs=argparse.REMAINDER)

    world = subcommands.add_parser(
        "world",
        add_help=False,
        help="build/serve an explorable 3D world from a photo (see `world -h`)",
    )
    world.add_argument("arguments", nargs=argparse.REMAINDER)

    subcommands.add_parser("formats", help="list supported local media formats")
    subcommands.add_parser("doctor", help="check dependencies and local data readiness")

    data = subcommands.add_parser("data", help="build or inspect local evidence data")
    data.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="arguments for local-data-pipeline/build_local_data.py",
    )
    return parser


def _doctor() -> int:
    modules = (
        "cv2",
        "torch",
        "geoclip",
        "paddleocr",
        "yt_dlp",
        "open_clip",
        "faiss",
    )
    data_root = Path("local-data").resolve()
    checks = {
        "python": sys.version.split()[0],
        "ffmpeg": shutil.which("ffmpeg"),
        "modules": {
            name: importlib.util.find_spec(name) is not None for name in modules
        },
        "data": {
            "geonames": (data_root / "geonames" / "geonames.sqlite3").is_file(),
            "ocr_models": (data_root / "ocr-models").is_dir(),
            "landmark_gallery": (data_root / "landmarks" / "gallery").is_dir(),
        },
    }
    print(json.dumps(checks, indent=2))
    required_ready = (
        checks["ffmpeg"]
        and all(checks["modules"].values())
        and all(checks["data"].values())
    )
    return 0 if required_ready else 1


def _run_data(arguments: Sequence[str]) -> int:
    script = REPO_ROOT / "local-data-pipeline" / "build_local_data.py"
    forwarded = list(arguments) or ["status"]
    completed = subprocess.run(
        [sys.executable, str(script), *forwarded],
        cwd=REPO_ROOT,
        check=False,
    )
    return int(completed.returncode)


def run(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if not arguments:
        build_parser().print_help()
        return 2

    # Friendly convenience and backward compatibility:
    #   view-finder photo.jpg ...
    #   view-finder --input photo.jpg ...
    if arguments[0] == "world":
        from .world.cli import run as world

        return world(arguments[1:])
    if arguments[0] not in {"locate", "formats", "doctor", "data"} and arguments[0] not in {
        "-h",
        "--help",
        "--version",
    }:
        from .main import run as locate

        return locate(arguments)
    if arguments[0].startswith("-") and arguments[0] not in {"-h", "--help", "--version"}:
        from .main import run as locate

        return locate(arguments)
    if arguments[0] == "locate":
        from .main import run as locate

        return locate(arguments[1:])

    parser = build_parser()
    parsed = parser.parse_args(arguments)
    if parsed.command == "formats":
        from .main import run as locate

        return locate(["--list-formats"])
    if parsed.command == "doctor":
        return _doctor()
    if parsed.command == "data":
        return _run_data(parsed.arguments)
    parser.print_help()
    return 0


def main() -> None:
    raise SystemExit(run())
