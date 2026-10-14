"""``view-finder world``: build, serve and set up the World Engine."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from .depth import DEFAULT_ESTIMATOR, DEPTH_ANYTHING_SMALL, ESTIMATORS, DepthModelUnavailable
from .profiles import PROFILES

logger = logging.getLogger("viewfinder.world")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="view-finder world",
        description="Build an explorable 3D world from a single photo.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def depth_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--depth-model",
            choices=ESTIMATORS,
            default=DEFAULT_ESTIMATOR,
            help="depth estimator; 'heuristic' needs no model but is crude (default: %(default)s)",
        )
        sub.add_argument(
            "--depth-model-id",
            default=None,
            help=f"Hugging Face id for the Depth Anything checkpoint (default: {DEPTH_ANYTHING_SMALL}). "
            "Larger V2 checkpoints are CC-BY-NC licensed.",
        )
        sub.add_argument("--device", default="auto", help="auto | cpu | cuda (default: auto)")
        sub.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    build = commands.add_parser("build", help="build a world from a photo into .glb/.json(/.html)")
    build.add_argument("photo", type=Path, help="input still image")
    build.add_argument("--quality", choices=list(PROFILES), default="desktop", help="(default: %(default)s)")
    build.add_argument("--out", type=Path, default=None, help="output directory (default: beside the photo)")
    build.add_argument("--html", action="store_true", help="also write a standalone single-file HTML viewer")
    depth_options(build)

    serve = commands.add_parser("serve", help="serve the viewer + upload API (open it on your phone)")
    serve.add_argument("--host", default="127.0.0.1", help="bind address; 0.0.0.0 to allow phones on your LAN")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--store", type=Path, default=Path("local-data") / "worlds", help="where worlds are kept")
    serve.add_argument("--max-worlds", type=int, default=30, help="keep only the newest N worlds")
    serve.add_argument("--max-upload-mb", type=int, default=25)
    serve.add_argument("--quality", choices=list(PROFILES), default="mobile",
                       help="profile used when the client does not ask for one")
    serve.add_argument("--token", default=None, help="fixed access token (default: random when off-loopback)")
    serve.add_argument("--no-token", action="store_true",
                       help="disable the access token even on a LAN address (not recommended)")
    serve.add_argument("--tls-cert", type=Path, default=None,
                       help="PEM certificate; HTTPS enables Tilt on phones (browsers block motion on plain HTTP)")
    serve.add_argument("--tls-key", type=Path, default=None, help="PEM private key for --tls-cert")
    depth_options(serve)

    setup = commands.add_parser("setup", help="download the depth model once (~100 MB, Apache-2.0)")
    setup.add_argument("--depth-model-id", default=DEPTH_ANYTHING_SMALL)
    setup.add_argument("-v", "--verbose", action="store_true")
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if not verbose:
        for noisy in ("transformers", "urllib3", "huggingface_hub", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _load(args: argparse.Namespace):
    from .depth import load_estimator

    return load_estimator(args.depth_model, device=args.device, model_id=args.depth_model_id)


def _build(args: argparse.Namespace) -> int:
    from .export import write_world
    from .ingest import ImageInputError
    from .pipeline import build_world

    if not args.photo.is_file():
        print(f"error: input file not found: {args.photo}", file=sys.stderr)
        return 2
    try:
        estimator = _load(args)
    except DepthModelUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    def progress(stage: str, fraction: float) -> None:
        if stage != "start":
            logger.debug("%-12s %3.0f%%", stage, fraction * 100)

    try:
        result = build_world(args.photo, profile=args.quality, estimator=estimator, progress=progress)
    except ImageInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        logger.debug("build failed", exc_info=True)
        print(f"error: world build failed: {exc}", file=sys.stderr)
        return 1
    output_dir = args.out or args.photo.resolve().parent
    paths = write_world(result, output_dir, args.photo.stem, html=args.html)
    manifest = result.manifest
    layers = ", ".join(f"{name} {stats['triangles']:,}" for name, stats in manifest["layers"].items())
    print(
        f"World built ({manifest['profile']}, {manifest['depth']['estimator']}) in "
        f"{manifest['timings_ms']['total'] / 1000:.1f}s: {len(result.glb) / 1e6:.1f} MB"
    )
    print(f"  triangles: {layers}")
    for warning in manifest["warnings"]:
        print(f"  note: {warning}")
    for kind, path in paths.items():
        print(f"  {kind:<8} {path}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .server import ServeConfig, create_server, server_urls

    if bool(args.tls_cert) != bool(args.tls_key):
        print("error: --tls-cert and --tls-key must be given together", file=sys.stderr)
        return 2
    try:
        estimator = _load(args)
    except DepthModelUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    config = ServeConfig(
        host=args.host,
        port=args.port,
        store=args.store,
        max_worlds=args.max_worlds,
        max_upload_mb=args.max_upload_mb,
        default_quality=args.quality,
        token=args.token,
        require_token=False if args.no_token else None,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
    )
    try:
        server = create_server(config, estimator)
    except OSError as exc:
        print(f"error: cannot listen on {args.host}:{args.port} ({exc})", file=sys.stderr)
        return 1
    print("View Finder World Engine is running. Open on your phone or browser:")
    for url in server_urls(server, config):
        print(f"  {url}")
    if server.token:
        print(f"Access token: {server.token} (needed once per device; keep it to yourself)")
    elif args.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: no access token: anyone on this network can use the engine.")
    if args.host in ("127.0.0.1", "localhost", "::1"):
        print("Only this computer can connect. Use --host 0.0.0.0 to allow a phone on your Wi-Fi.")
    print(f"Depth: {estimator.describe()['estimator']} on {estimator.describe()['device']}. Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.runner.stop()
        server.server_close()
    return 0


def _setup(args: argparse.Namespace) -> int:
    from .depth import download_model, model_cached

    if model_cached(args.depth_model_id):
        print(f"Depth model {args.depth_model_id} is already installed.")
        return 0
    print(f"Downloading {args.depth_model_id} (~100 MB) into the Hugging Face cache…")
    try:
        paths = download_model(args.depth_model_id)
    except Exception as exc:
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"installed": paths}, indent=2))
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    _configure_logging(getattr(args, "verbose", False))
    return {"build": _build, "serve": _serve, "setup": _setup}[args.command](args)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
