"""Command-line entry point for the View Finder geolocation pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO, TypeVar


logger = logging.getLogger("viewfinder")

# Accepted formats live in one module so the CLI can never drift from what the
# image loader supports. Imported both ways because main.py runs as a module
# (`python -m viewfinder.main`) and as a script (`python viewfinder/main.py`).
try:
    from .formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    from .progress import TerminalProgressBar
except ImportError:  # pragma: no cover - script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from viewfinder.formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    from viewfinder.progress import TerminalProgressBar
DEFAULT_FPS = 1.0
DEFAULT_TOP_K = 5
DEFAULT_MAX_PREDICTIONS = 5

# GeoCLIP is the default visual predictor. The retrieval and ensemble predictors
# that previously shared this slot are shelved in experiments/retrieval/ -- they
# worked, but only inside their gallery's footprint, and contributed nothing on
# globally diverse imagery. See "Retrieval and ensemble" in README.md.
DEFAULT_PREDICTOR = "geoclip"

_T = TypeVar("_T")

# label -> a context manager exposing .update(done, total). The CLI supplies
# terminal bars; the desktop GUI supplies widget-updating adapters.
ProgressFactory = Callable[[str], Any]


def _terminal_progress_factory(*, quiet: bool, verbose: bool) -> ProgressFactory:
    """Build progress bars for terminal use.

    Bars are suppressed under --quiet (which prints only the machine-readable
    result) and under --verbose, where per-batch debug logging is already the
    progress indication and would shred a carriage-return bar.
    """
    enabled = not quiet and not verbose

    def make(label: str) -> TerminalProgressBar:
        return TerminalProgressBar(label, enabled=enabled)

    return make


class PipelineError(RuntimeError):
    """A pipeline-stage failure suitable for showing to a CLI user."""


class UserInputError(ValueError):
    """An invalid CLI value discovered after argument parsing."""


def build_parser() -> argparse.ArgumentParser:
    """Build and return the View Finder argument parser."""
    parser = argparse.ArgumentParser(
        prog="view-finder",
        description=(
            "Estimate where a landscape or travel photo/video was captured "
            "using embedded metadata, landmark/text/web evidence, and visual models."
        ),
    )
    parser.add_argument(
        "source",
        nargs="?",
        metavar="SOURCE",
        help="local image/video path or public Instagram/TikTok video URL",
    )
    parser.add_argument(
        "--input",
        metavar="SOURCE",
        help="backward-compatible alias for positional SOURCE",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        metavar="FLOAT",
        help="video sampling rate in frames per second (default: %(default)s)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=24,
        metavar="INT",
        help="maximum retained video frames to analyze (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="INT",
        help="number of ranked candidates to retain/report (default: %(default)s)",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=DEFAULT_MAX_PREDICTIONS,
        metavar="INT",
        help="maximum distinct location predictions to return (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="JSON result path (default: INPUT_STEM_viewfinder.json beside input)",
    )
    parser.add_argument(
        "--download-dir",
        default=os.getenv("VIEWFINDER_DOWNLOAD_DIR", "local-data/social-video-cache"),
        metavar="PATH",
        help=(
            "cache directory for social video downloads "
            "(default: %(default)s; env: VIEWFINDER_DOWNLOAD_DIR)"
        ),
    )
    parser.add_argument(
        "--max-video-seconds",
        type=int,
        default=600,
        metavar="INT",
        help="reject downloaded social clips longer than this (default: %(default)s)",
    )
    parser.add_argument(
        "--max-download-mb",
        type=int,
        default=500,
        metavar="INT",
        help="reject social video downloads larger than this (default: %(default)s)",
    )
    parser.add_argument(
        "--predictor",
        choices=(
            "geoclip",
            "hybrid",
            "cloud-hybrid",
            "plonk-osv",
            "plonk-yfcc",
            "trusted",
        ),
        default=None,
        help=(
            "which geolocation system to run: geoclip (default) and PLONK are visual "
            "models; hybrid uses fully local EXIF/OCR/GeoNames/Commons evidence and is "
            "fail-closed; cloud-hybrid is the optional Google implementation; "
            f"trusted runs GeoCLIP plus both PLONK models (default: {DEFAULT_PREDICTOR})"
        ),
    )
    parser.add_argument(
        "--online-evidence",
        action="store_true",
        help=(
            "with --predictor cloud-hybrid, upload the image to Google Cloud Vision and "
            "resolve OCR place text with Google Places; may incur API charges"
        ),
    )
    parser.add_argument(
        "--require-evidence",
        action="store_true",
        help=(
            "with --predictor hybrid/cloud-hybrid, exit 3 unless evidence is strong enough for "
            "automatic use (embedded GPS, attributed near-duplicate, or corroborated evidence)"
        ),
    )
    parser.add_argument(
        "--local-data",
        default=os.getenv("VIEWFINDER_LOCAL_DATA", "local-data"),
        metavar="PATH",
        help=(
            "local GeoNames/OCR/Commons/OSM data root used by --predictor hybrid "
            "(default: %(default)s; env: VIEWFINDER_LOCAL_DATA)"
        ),
    )
    parser.add_argument(
        "--gallery",
        metavar="PATH",
        help=(
            "custom GPS gallery directory or metadata file built by gallery-pipeline "
            "(default: GeoCLIP's bundled gallery; env: VIEWFINDER_GALLERY)"
        ),
    )
    parser.add_argument(
        "--gallery-mode",
        choices=("augment", "replace"),
        help=(
            "augment adds the custom coordinates to GeoCLIP's bundled gallery; "
            "replace restricts predictions to the custom coordinates only "
            "(default: augment; env: VIEWFINDER_GALLERY_MODE)"
        ),
    )
    parser.add_argument(
        "--plonk-samples",
        type=int,
        default=32,
        metavar="INT",
        help="posterior samples per image for each PLONK model (default: %(default)s)",
    )
    parser.add_argument(
        "--trust-radius-km",
        type=float,
        default=200.0,
        metavar="FLOAT",
        help="maximum all-pairs model separation for trusted consensus (default: %(default)s)",
    )
    parser.add_argument(
        "--required-success-rate",
        type=float,
        default=0.95,
        metavar="FLOAT",
        help="minimum held-out Wilson lower bound required to emit (default: %(default)s)",
    )
    parser.add_argument(
        "--trust-success-km",
        type=float,
        default=200.0,
        metavar="FLOAT",
        help="distance defining calibration success (default: %(default)s)",
    )
    parser.add_argument(
        "--trust-calibration",
        metavar="PATH",
        help="held-out calibration JSON (default: models/trust_calibration.json)",
    )
    parser.add_argument(
        "--require-trust",
        action="store_true",
        help="with --predictor trusted, exit 3 when the trust gate abstains",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", "-v", action="store_true", help="enable debug logging for every stage"
    )
    verbosity.add_argument(
        "--quiet", "-q", action="store_true", help="suppress progress; print only final result"
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="print every accepted image and video extension, then exit",
    )
    parser.add_argument(
        "--use-stock-geoclip",
        action="store_true",
        help=(
            "use GeoCLIP's original location encoder instead of the bundled "
            "fine-tuned one. The fine-tuned encoder scores 953 km median error on "
            "the independent benchmark against stock's 1331 km, so this is for "
            "comparison and debugging rather than normal use"
        ),
    )
    parser.add_argument(
        "--crop-screen",
        choices=("auto", "off"),
        default="auto",
        help=(
            "when the input is a photo *of* a display or framed picture, crop to the "
            "screen's content before predicting, so the model judges the scene rather "
            "than the room around it. auto (default) only acts on a confident "
            "detection and leaves ordinary photographs untouched; off disables it "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--remove-text",
        action="store_true",
        help=(
            "inpaint overlaid text (watermarks, captions, timestamps) before "
            "predicting. OFF by default: real scene text such as street signs is a "
            "strong geolocation cue, so removing it can REDUCE accuracy. Downloads "
            "EAST weights (~96 MB) once."
        ),
    )
    return parser


def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    """Configure CLI logging without changing the pipeline modules."""
    level = logging.CRITICAL if quiet else logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)


def _resolve_remote_source(args: argparse.Namespace) -> dict[str, Any] | None:
    """Download an allowed public social URL and replace it with its cache path."""
    raw_source = args.input or args.source
    if not raw_source or not str(raw_source).lower().startswith(("http://", "https://")):
        return None
    download_social_video = _stage(
        "Loading social-video URL support",
        lambda: _import_attr("media_source", "download_social_video"),
        hint="Install viewfinder/requirements-social-video.txt and put FFmpeg on PATH.",
    )
    resolved = _stage(
        "Downloading public social video",
        lambda: download_social_video(
            str(raw_source),
            download_dir=args.download_dir,
            max_duration_seconds=args.max_video_seconds,
            max_download_bytes=args.max_download_mb * 1024 * 1024,
        ),
        hint=(
            "Only public Instagram/TikTok posts are supported. Private, login-gated, "
            "deleted, or platform-blocked posts cannot be fetched."
        ),
    )
    args.input = str(resolved.local_path)
    args.source = None
    if not args.output:
        args.output = str(
            Path.cwd() / f"{resolved.local_path.stem}_viewfinder.json"
        )
    return dict(resolved.source)


def _validate_args(args: argparse.Namespace) -> tuple[Path, str, Path]:
    """Validate cheap input facts before importing or initializing ML code."""
    if args.input and args.source:
        raise UserInputError("provide SOURCE or --input, not both")
    raw_source = args.input or args.source
    if not raw_source:
        raise UserInputError("a local media path or Instagram/TikTok URL is required")
    input_path = Path(raw_source).expanduser().resolve()
    if not input_path.exists():
        raise UserInputError(f"input file does not exist: {input_path}")
    if not input_path.is_file():
        raise UserInputError(f"input path is not a file: {input_path}")

    extension = input_path.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        input_type = "image"
    elif extension in VIDEO_EXTENSIONS:
        input_type = "video"
    else:
        # The accepted set is large (it tracks Pillow's decoders), so lead with
        # the common formats and point at --list-formats for the full list.
        common = ".jpg, .jpeg, .png, .webp, .tif/.tiff, .bmp, .gif, .avif"
        raise UserInputError(
            f"unsupported input format '{extension or '(none)'}'. "
            f"Common image formats: {common}. Videos: .mp4, .mov, .avi, .mkv, .webm. "
            f"Run with --list-formats to see every accepted extension."
        )

    if input_type == "video" and (not math.isfinite(args.fps) or args.fps <= 0):
        raise UserInputError("--fps must be a finite number greater than zero")
    if args.max_frames <= 0:
        raise UserInputError("--max-frames must be greater than zero")
    if args.top_k <= 0:
        raise UserInputError("--top-k must be greater than zero")
    if args.max_predictions <= 0:
        raise UserInputError("--max-predictions must be greater than zero")
    if args.plonk_samples <= 0:
        raise UserInputError("--plonk-samples must be greater than zero")
    if args.max_video_seconds <= 0:
        raise UserInputError("--max-video-seconds must be greater than zero")
    if args.max_download_mb <= 0:
        raise UserInputError("--max-download-mb must be greater than zero")
    if not math.isfinite(args.trust_radius_km) or args.trust_radius_km <= 0:
        raise UserInputError("--trust-radius-km must be a finite number greater than zero")
    if not 0.0 < args.required_success_rate <= 1.0:
        raise UserInputError("--required-success-rate must be in (0, 1]")
    if not math.isfinite(args.trust_success_km) or args.trust_success_km <= 0:
        raise UserInputError("--trust-success-km must be a finite number greater than zero")
    if args.require_trust and args.predictor != "trusted":
        raise UserInputError("--require-trust is only valid with --predictor trusted")
    if args.online_evidence and args.predictor != "cloud-hybrid":
        raise UserInputError(
            "--online-evidence is only valid with --predictor cloud-hybrid"
        )
    if args.require_evidence and args.predictor not in ("hybrid", "cloud-hybrid"):
        raise UserInputError(
            "--require-evidence is only valid with --predictor hybrid or cloud-hybrid"
        )
    if input_type == "video" and args.predictor == "cloud-hybrid":
        raise UserInputError(
            "cloud-hybrid accepts images only; use the fully local hybrid or "
            "select a visual predictor for video"
        )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_viewfinder.json")

    if output_path == input_path:
        raise UserInputError("--output must not overwrite the input file")

    return input_path, input_type, output_path


def _import_attr(module: str, attribute: str) -> Any:
    """Import a pipeline attribute lazily, supporting module and script use."""
    package = __package__ or "viewfinder"
    if __package__ in {None, ""}:
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    return getattr(importlib.import_module(f"{package}.{module}"), attribute)


def _stage(name: str, operation: Callable[[], _T], *, hint: str = "") -> _T:
    """Run one stage and turn internal exceptions into actionable CLI errors."""
    logger.info("%s...", name)
    try:
        value = operation()
    except PipelineError:
        raise
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        suffix = f" {hint}" if hint else ""
        raise PipelineError(f"{name} failed: {detail}.{suffix}") from exc
    logger.debug("%s complete", name)
    return value


def _new_geolocator(use_stock_geoclip: bool = False) -> Any:
    """Construct GeoCLIP in enforced offline mode.

    Any custom gallery selection is read from the environment, which the CLI
    flags populate in :func:`run`. This keeps the construction signature stable
    for both the image and video pipelines.

    By default this loads the fine-tuned location encoder; ``use_stock_geoclip``
    keeps GeoCLIP's original weights for comparison.
    """
    geolocator_class = _import_attr("geolocator", "GeoLocator")
    return geolocator_class(offline=True, use_stock_geoclip=use_stock_geoclip)


def _new_plonk_geolocator(
    model_id: str,
    samples_per_image: int,
    *,
    offline: bool = False,
) -> Any:
    """Construct one optional PLONK predictor without importing it at startup."""
    geolocator_class = _import_attr("plonk_geolocator", "PlonkGeolocator")
    return geolocator_class(
        model_id,
        offline=offline,
        samples_per_image=samples_per_image,
    )


def _new_cloud_hybrid_geolocator(
    input_path: Path,
    *,
    online_evidence: bool,
    samples_per_image: int,
) -> Any:
    """Construct the evidence-first image predictor with PLONK as weak fallback."""
    geolocator_class = _import_attr("hybrid_geolocator", "HybridGeolocator")

    def build_visual_fallback() -> Any:
        try:
            return _new_plonk_geolocator(
                "nicolas-dufour/PLONK_OSV_5M",
                samples_per_image,
                offline=True,
            )
        except Exception as exc:
            logger.warning(
                "PLONK fallback is unavailable (%s); using cached GeoCLIP instead.",
                exc,
            )
            return _new_geolocator()

    return geolocator_class(
        input_path,
        online_evidence=online_evidence,
        visual_fallback_factory=build_visual_fallback,
        visual_fallback_name="PLONK OSV-5M (GeoCLIP if PLONK is unavailable)",
    )


def _new_local_hybrid_geolocator(
    input_path: Path,
    *,
    data_root: Path,
    samples_per_image: int,
) -> Any:
    """Construct the fully local evidence system with PLONK as a weak prior."""
    geolocator_class = _import_attr("local_hybrid_geolocator", "LocalHybridGeolocator")

    def build_visual_prior() -> Any:
        try:
            return _new_plonk_geolocator(
                "nicolas-dufour/PLONK_OSV_5M",
                samples_per_image,
                offline=True,
            )
        except Exception as exc:
            logger.warning(
                "PLONK prior is unavailable (%s); using cached GeoCLIP as an "
                "unauthorized visual hypothesis.",
                exc,
            )
            return _new_geolocator()

    return geolocator_class(
        input_path,
        data_root=data_root,
        visual_prior_factory=build_visual_prior,
        visual_prior_name="PLONK OSV-5M (GeoCLIP if PLONK is unavailable)",
    )


@dataclass(frozen=True)
class PredictorSpec:
    """How to build and label one predictor.

    The predictors are independent systems that happen to share the
    aggregation and reverse-geocoding stages. Describing them uniformly lets the
    pipelines run either one without branching on predictor type, while keeping
    their results strictly separate.
    """

    key: str
    label: str
    build: Callable[[], Any]
    load_hint: str


def _predictor_specs(args: argparse.Namespace) -> list[PredictorSpec]:
    """Return the predictors selected by --predictor, in report order."""
    geoclip = PredictorSpec(
        key="geoclip",
        label="GeoCLIP",
        build=lambda: _new_geolocator(use_stock_geoclip=args.use_stock_geoclip),
        load_hint=(
            "Install the requirements and pre-cache the GeoCLIP weights; "
            "the CLI does not access the network."
        ),
    )
    plonk_osv = PredictorSpec(
        key="plonk-osv",
        label="PLONK OSV-5M",
        build=lambda: _new_plonk_geolocator(
            "nicolas-dufour/PLONK_OSV_5M", args.plonk_samples
        ),
        load_hint=(
            "Install the optional PLONK dependencies documented in README.md. "
            "The first run also downloads the official model and feature backbone."
        ),
    )
    plonk_yfcc = PredictorSpec(
        key="plonk-yfcc",
        label="PLONK YFCC",
        build=lambda: _new_plonk_geolocator(
            "nicolas-dufour/PLONK_YFCC", args.plonk_samples
        ),
        load_hint=(
            "Install the optional PLONK dependencies documented in README.md. "
            "The first run also downloads the official model and feature backbone."
        ),
    )
    hybrid = PredictorSpec(
        key="hybrid",
        label="fully local evidence hybrid",
        build=lambda: _new_local_hybrid_geolocator(
            Path(args.input).expanduser().resolve(),
            data_root=Path(args.local_data).expanduser().resolve(),
            samples_per_image=args.plonk_samples,
        ),
        load_hint=(
            "Install viewfinder/requirements-local-evidence.txt and build local data "
            "with local-data-pipeline/build_local_data.py. PLONK remains optional."
        ),
    )
    cloud_hybrid = PredictorSpec(
        key="cloud-hybrid",
        label="cloud evidence hybrid",
        build=lambda: _new_cloud_hybrid_geolocator(
            Path(args.input).expanduser().resolve(),
            online_evidence=args.online_evidence,
            samples_per_image=args.plonk_samples,
        ),
        load_hint=(
            "Install viewfinder/requirements-hybrid.txt and configure Google "
            "credentials for the optional cloud evidence path."
        ),
    )

    if args.predictor == "hybrid":
        return [hybrid]
    if args.predictor == "cloud-hybrid":
        return [cloud_hybrid]
    if args.predictor == "geoclip":
        return [geoclip]
    if args.predictor == "plonk-osv":
        return [plonk_osv]
    if args.predictor == "plonk-yfcc":
        return [plonk_yfcc]
    if args.predictor == "trusted":
        return [geoclip, plonk_osv, plonk_yfcc]
    return [geoclip]


def _candidate_dict(candidate: Any) -> dict[str, float]:
    """Serialize a GeoCLIP candidate without depending on dataclass internals."""
    return {
        "latitude": round(float(candidate.lat), 6),
        "longitude": round(float(candidate.lon), 6),
        "score": float(candidate.score),
    }


def _location_dict(lat: float, lon: float, place: dict[str, str]) -> dict[str, Any]:
    """Build the common location portion of an output prediction."""
    return {
        "latitude": round(float(lat), 6),
        "longitude": round(float(lon), 6),
        "country": place["country"],
        "admin1": place["admin1"],
        "city": place["city"],
    }


def _serialize_aggregate(aggregate_result: Any, places: list[dict[str, str]]) -> dict[str, Any]:
    """Build backward-compatible primary output plus all ranked predictions."""
    ranked: list[dict[str, Any]] = []
    for rank, (cluster_prediction, place) in enumerate(
        zip(aggregate_result.predictions, places),
        start=1,
    ):
        ranked.append(
            {
                "rank": rank,
                "lat": round(float(cluster_prediction.lat), 6),
                "lon": round(float(cluster_prediction.lon), 6),
                "confidence": round(float(cluster_prediction.confidence), 1),
                "support_count": int(cluster_prediction.support_count),
                "candidate_count": int(cluster_prediction.candidate_count),
                "top_similarity": round(float(cluster_prediction.top_similarity), 4),
                "country": place["country"],
                "admin1": place["admin1"],
                "city": place["city"],
                "confidence_details": cluster_prediction.confidence_breakdown.as_dict(),
            }
        )

    if not ranked:
        raise PipelineError("Aggregation produced no location predictions.")

    best = ranked[0]
    primary = _location_dict(best["lat"], best["lon"], best)
    primary.update(
        {
            "confidence_percent": best["confidence"],
            "support_count": best["support_count"],
            "candidate_count": best["candidate_count"],
            "top_similarity": best["top_similarity"],
        }
    )
    return {
        "prediction": primary,
        "predictions": ranked,
        "alternatives": ranked[1:],
    }


def _process_image(
    input_path: Path,
    top_k: int,
    max_predictions: int,
    spec: PredictorSpec,
    remove_text_enabled: bool = False,
    crop_screen: str = "auto",
) -> dict[str, Any]:
    """Run image prediction, multi-hypothesis clustering, and place lookup."""
    load_image = _stage(
        "Loading image support",
        lambda: _import_attr("video_processor", "load_image"),
        hint="Install dependencies with 'python -m pip install -r viewfinder/requirements.txt'.",
    )
    image = _stage(
        "Loading image",
        lambda: load_image(str(input_path)),
        hint=(
            f"'{input_path.name}' could not be decoded. The extension is recognised, so "
            "the file itself is most likely truncated, corrupt, or actually a different "
            "format than its name suggests. Run --list-formats for every accepted "
            "extension."
        ),
    )
    # Cropping runs before text removal: once the frame is reduced to the
    # screen's content, any remaining overlay text is the only text left, and
    # the room's signage is already gone.
    screen_crop_info: dict[str, Any] | None = None
    if crop_screen != "off":
        detect_and_crop = _stage(
            "Loading screen-detection support",
            lambda: _import_attr("screen_crop", "detect_and_crop"),
            hint="Screen detection needs OpenCV; reinstall viewfinder/requirements.txt.",
        )
        crop_result = _stage(
            "Checking whether the photo is of a screen",
            lambda: detect_and_crop(image),
            hint="Pass --crop-screen off to skip this check.",
        )
        screen_crop_info = crop_result.as_dict()
        if crop_result.cropped:
            image = crop_result.image
        else:
            logger.debug("screen crop: no crop applied (%s)", crop_result.reason)

    text_removal_info: dict[str, Any] | None = None
    if remove_text_enabled:
        remove_text = _stage(
            "Loading text-removal support",
            lambda: _import_attr("text_removal", "remove_text"),
            hint="Text removal needs OpenCV; reinstall viewfinder/requirements.txt.",
        )
        result = _stage(
            "Removing overlaid text",
            lambda: remove_text(image),
            hint="Text removal downloads EAST weights (~96 MB) once; check connectivity.",
        )
        text_removal_info = result.as_dict()
        image = result.image

    geolocator = _stage(
        f"Loading {spec.label} predictor",
        spec.build,
        hint=spec.load_hint,
    )
    candidates = _stage(
        f"Analyzing image with {spec.label}",
        lambda: geolocator.predict_image(image, top_k=top_k),
        hint="Try a smaller image or verify that the selected Torch device has enough memory.",
    )
    if not candidates:
        raise PipelineError(
            f"Analyzing image failed: {spec.label} returned no location candidates."
        )

    aggregate = _stage(
        "Loading aggregation support",
        lambda: _import_attr("aggregator", "aggregate"),
    )
    frame_prediction_class = _stage(
        "Loading prediction support",
        lambda: _import_attr("aggregator", "FramePrediction"),
    )
    aggregate_result = _stage(
        "Clustering image candidates",
        lambda: aggregate(
            [frame_prediction_class(candidates=candidates)],
            top_k=top_k,
            max_predictions=max_predictions,
        ),
    )

    reverse_geocode_batch = _stage(
        "Loading offline place index",
        lambda: _import_attr("reverse_geocode", "reverse_geocode_batch"),
        hint="Install reverse_geocoder from viewfinder/requirements.txt.",
    )
    places = _stage(
        "Finding nearest named places",
        lambda: reverse_geocode_batch(
            [(prediction.lat, prediction.lon) for prediction in aggregate_result.predictions]
        ),
        hint="Reinstall reverse_geocoder if its bundled dataset is unavailable.",
    )

    output = _serialize_aggregate(aggregate_result, places)
    output["predictor"] = spec.key
    output["processing"] = {
        "candidates_returned": len(candidates),
        "clusters_returned": len(aggregate_result.predictions),
    }
    if screen_crop_info is not None:
        output["processing"]["screen_crop"] = screen_crop_info
    if text_removal_info is not None:
        output["processing"]["text_removal"] = text_removal_info
    _attach_model_provenance(output, geolocator)
    _attach_evidence_assessment(output, geolocator)
    return output


def _attach_model_provenance(output: dict[str, Any], geolocator: Any) -> None:
    """Record model and training-data provenance when a predictor exposes it."""
    provenance = getattr(geolocator, "provenance", None)
    if provenance is not None and hasattr(provenance, "as_dict"):
        output["model"] = provenance.as_dict()


def _attach_evidence_assessment(output: dict[str, Any], geolocator: Any) -> None:
    """Expose hybrid evidence and prevent visual-only candidates becoming an answer."""
    assessment = getattr(geolocator, "last_assessment", None)
    if assessment is None or not hasattr(assessment, "as_dict"):
        return

    evidence = assessment.as_dict()
    output["evidence_assessment"] = evidence
    output["warning"] = evidence["warning"]

    if assessment.decision == "abstain":
        output["visual_hypotheses"] = output.get("predictions", [])
        output["prediction"] = None
        output["predictions"] = []
        output["alternatives"] = []
        return

    prediction = output.get("prediction")
    if isinstance(prediction, dict):
        # The generic aggregator's confidence is intentionally not shown here:
        # with one evidence coordinate it is a hand-built concentration score,
        # not a calibrated probability that the location is correct.
        for key in ("confidence_percent", "support_count", "candidate_count", "top_similarity"):
            prediction.pop(key, None)
        prediction["evidence_tier"] = assessment.reliability_tier
        prediction["evidence_source"] = assessment.selected_source
        prediction["automation_accepted"] = assessment.automation_accepted


def _process_video(
    input_path: Path,
    fps: float,
    max_frames: int,
    max_duration_seconds: int,
    top_k: int,
    max_predictions: int,
    progress_factory: ProgressFactory,
    spec: PredictorSpec,
) -> dict[str, Any]:
    """Run frame extraction, batched inference, aggregation, and place lookup."""
    extract_frames = _stage(
        "Loading video support",
        lambda: _import_attr("video_processor", "extract_frames"),
        hint="Install OpenCV and the other packages in viewfinder/requirements.txt.",
    )
    with progress_factory("Sampling frames") as bar:
        frames, stats = _stage(
            f"Sampling video at {fps:g} fps and filtering blurry frames",
            lambda: extract_frames(
                str(input_path),
                fps=fps,
                max_frames=max_frames,
                progress=bar.update,
            ),
            hint=(
                "Confirm the file is a complete, readable video that OpenCV can decode. "
                "A truncated download or an unsupported codec both look like this."
            ),
        )
    if not frames:
        raise PipelineError(
            f"No usable frames survived filtering: all {stats.sampled} sampled frame(s) "
            "were rejected as too blurry or as near-duplicates of each other.\n"
            "  Try one of:\n"
            "    --fps 2          sample more often (currently "
            f"{fps:g}/second of video)\n"
            "    --max-frames N   raise the retained-frame cap\n"
            "  A static shot or a heavily motion-blurred clip is the usual cause."
        )
    logger.info(
        "Retained %d/%d sampled frames; starting batched inference",
        len(frames),
        stats.sampled,
    )

    geolocator = _stage(
        f"Loading {spec.label} predictor",
        spec.build,
        hint=spec.load_hint,
    )
    with progress_factory(f"{spec.label} inference") as bar:
        predictions = _stage(
            f"Processing {len(frames)} video frames with {spec.label}",
            lambda: geolocator.predict_frames(frames, top_k=top_k, progress=bar.update),
            hint=(
                "Try a lower --fps or --max-frames; the selected Torch device may not "
                "have enough memory for this many frames."
            ),
        )
    if not predictions:
        raise PipelineError(f"{spec.label} inference produced no frame predictions.")
    if any(not prediction.candidates for prediction in predictions):
        raise PipelineError(
            f"{spec.label} inference returned an empty candidate list for a frame."
        )
    evidence_predictions = getattr(geolocator, "evidence_frame_predictions", None)
    aggregation_predictions = (
        evidence_predictions if evidence_predictions else predictions
    )
    if (
        stats.duration_seconds
        and stats.duration_seconds > max_duration_seconds
    ):
        raise UserInputError(
            f"video duration {stats.duration_seconds:.1f}s exceeds the "
            f"{max_duration_seconds}s limit; raise --max-video-seconds intentionally"
        )

    aggregate = _stage(
        "Loading aggregation support",
        lambda: _import_attr("aggregator", "aggregate"),
    )
    aggregate_result = _stage(
        "Clustering frame predictions",
        lambda: aggregate(
            aggregation_predictions,
            top_k=top_k,
            max_predictions=max_predictions,
        ),
        hint="Use a clip with multiple clear views of the same general area.",
    )

    reverse_geocode_batch = _stage(
        "Loading offline place index",
        lambda: _import_attr("reverse_geocode", "reverse_geocode_batch"),
        hint="Install reverse_geocoder from viewfinder/requirements.txt.",
    )
    places = _stage(
        "Finding nearest named places",
        lambda: reverse_geocode_batch(
            [(prediction.lat, prediction.lon) for prediction in aggregate_result.predictions]
        ),
        hint="Reinstall reverse_geocoder if its bundled dataset is unavailable.",
    )

    processing = stats.as_dict()
    processing.update(
        {
            "frames_analyzed": len(predictions),
            "frame_dispersion_km": round(float(aggregate_result.frame_dispersion_km), 3),
            "confidence_details": aggregate_result.confidence.as_dict(),
            "clusters_returned": len(aggregate_result.predictions),
        }
    )
    if evidence_predictions is not None:
        processing["evidence_frames"] = len(evidence_predictions)
    output = _serialize_aggregate(aggregate_result, places)
    output["predictor"] = spec.key
    output["processing"] = processing
    _attach_model_provenance(output, geolocator)
    _attach_evidence_assessment(output, geolocator)
    return output


def _write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Create the output directory if needed and write formatted UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _print_json(value: Any) -> None:
    """Print JSON on legacy Windows consoles without crashing on Unicode."""
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    ensure_ascii = not encoding.startswith(("utf-8", "utf8"))
    print(json.dumps(value, ensure_ascii=ensure_ascii, allow_nan=False))


def _place_label(prediction: dict[str, Any]) -> str:
    """Format non-empty reverse-geocoded place components."""
    parts = [prediction.get("city"), prediction.get("admin1"), prediction.get("country")]
    return ", ".join(str(part) for part in parts if part) or "nearest known place unavailable"


PREDICTOR_TITLES = {
    "hybrid": "Fully local hybrid (OCR, GeoNames/OSM, Commons, visual abstention)",
    "cloud-hybrid": "Cloud evidence hybrid (optional Google services)",
    "geoclip": "GeoCLIP (learned location embeddings; no reference photos)",
    "plonk-osv": "PLONK OSV-5M (probabilistic street-view model)",
    "plonk-yfcc": "PLONK YFCC (probabilistic general-photo model)",
}

SUMMARY_WIDTH = 66


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full result to the handful of fields worth showing a human.

    Shared by the CLI summary and the desktop GUI so the two can never drift.
    Returns a dict with ``place``, ``latitude``, ``longitude``, ``confidence``
    (may be None), ``predictor``, ``warning``
    (may be None), and ``alternatives``.
    """
    predictor = result.get("predictor") or ", ".join(result.get("predictors", [])) or "unknown"
    prediction = result.get("prediction")

    # Three genuinely different outcomes, which must not be conflated:
    #   single     -- one predictor, one answer;
    #   comparison -- --predictor both, which deliberately produces no combined
    #                 answer, and is not an abstention;
    #   abstained  -- a predictor that could have answered declined to.
    if "results" in result and "prediction" not in result:
        mode = "comparison"
    elif prediction is None:
        mode = "abstained"
    else:
        mode = "single"

    summary: dict[str, Any] = {
        "mode": mode,
        "predictor": predictor,
        "predictor_title": PREDICTOR_TITLES.get(predictor, predictor),
        "warning": result.get("warning"),
        "place": None,
        "latitude": None,
        "longitude": None,
        "confidence": None,
        "alternatives": [],
        "per_predictor": [],
        "abstained": mode == "abstained",
        "crop_note": None,
    }

    # An automatic crop changes what was actually analysed, so it is always
    # reported rather than left to the JSON.
    screen_crop = (result.get("processing") or {}).get("screen_crop") or {}
    if screen_crop.get("cropped"):
        summary["crop_note"] = (
            "cropped to a detected screen covering "
            f"{screen_crop['area_fraction'] * 100:.0f}% of the photo "
            "(pass --crop-screen off to disable)"
        )

    if mode == "comparison":
        for key, block in result["results"].items():
            block_prediction = block.get("prediction")
            if not block_prediction:
                continue
            summary["per_predictor"].append(
                {
                    "predictor": key,
                    "place": _place_label(block_prediction),
                    "latitude": block_prediction.get("latitude"),
                    "longitude": block_prediction.get("longitude"),
                    "confidence": block_prediction.get("confidence_percent"),
                }
            )
        return summary

    if prediction is None:
        # A fail-closed abstention still has something to say: the rejected
        # visual hypothesis, clearly labelled as not an answer.
        hypotheses = result.get("visual_hypotheses") or []
        if hypotheses:
            best = hypotheses[0]
            summary["place"] = _place_label(best)
            summary["latitude"] = best.get("lat")
            summary["longitude"] = best.get("lon")
        return summary

    summary["place"] = _place_label(prediction)
    summary["latitude"] = prediction.get("latitude")
    summary["longitude"] = prediction.get("longitude")
    summary["confidence"] = prediction.get("confidence_percent")
    summary["alternatives"] = [
        {
            "rank": alternative["rank"],
            "place": _place_label(alternative),
            "latitude": alternative["lat"],
            "longitude": alternative["lon"],
            "confidence": alternative["confidence"],
        }
        for alternative in result.get("predictions", [])[1:]
    ]
    return summary


def _print_summary(result: dict[str, Any], stream: TextIO) -> None:
    """Print the compact final summary, always, at every verbosity level.

    Deliberately the last thing written, so a long run's answer is never buried
    in stage logs. Under --quiet this goes to stderr, leaving the JSON on
    stdout byte-for-byte as scripts already expect.
    """
    summary = summarize(result)
    write = lambda text: print(text, file=stream)  # noqa: E731

    write("")
    write("=" * SUMMARY_WIDTH)
    if summary["mode"] == "comparison":
        write("  RESULT: separate predictors, not combined")
    elif summary["abstained"]:
        write("  RESULT: no location accepted (fail-closed abstention)")
    else:
        write(f"  RESULT: {summary['place']}")
    write("=" * SUMMARY_WIDTH)

    if summary["mode"] == "comparison":
        for entry in summary["per_predictor"]:
            confidence = (
                f"{entry['confidence']:.1f}%" if entry["confidence"] is not None else "n/a"
            )
            write(f"  {entry['predictor']:<12} {entry['place']}")
            write(
                f"  {'':<12} {entry['latitude']:.6f}, {entry['longitude']:.6f}"
                f"  ({confidence})"
            )
        write("=" * SUMMARY_WIDTH)
        return

    if summary["latitude"] is not None:
        coordinates = f"{summary['latitude']:.6f}, {summary['longitude']:.6f}"
        if summary["abstained"]:
            # Name the rejected hypothesis, clearly marked as not an answer.
            write(f"  Hypothesis   {summary['place']} — {coordinates}")
            write("               (manual review only; not an accepted result)")
        else:
            write(f"  Coordinates  {coordinates}")
    if summary["confidence"] is not None:
        write(f"  Confidence   {summary['confidence']:.1f}%")
    write(f"  Predictor    {summary['predictor']}")
    if summary["crop_note"]:
        write(f"  Preprocessing {summary['crop_note']}")
    if summary["warning"]:
        write(f"  ! {summary['warning']}")

    alternatives = summary["alternatives"]
    if alternatives:
        shown = ", ".join(
            f"{alternative['place']} ({alternative['confidence']:.0f}%)"
            for alternative in alternatives[:3]
        )
        write(f"  Alternatives {shown}")
    write("=" * SUMMARY_WIDTH)


def _print_result(result: dict[str, Any], output_path: Path, *, quiet: bool) -> None:
    """Print either a compact machine-readable result or a human summary."""
    if "results" in result:
        _print_comparison(result, output_path, quiet=quiet)
        return
    _print_single(result, output_path, quiet=quiet)


def _print_comparison(result: dict[str, Any], output_path: Path, *, quiet: bool) -> None:
    """Print each predictor and, when present, the evidence-gated trust decision."""
    if quiet:
        compact: dict[str, Any] = {
            "prediction": result.get("prediction"),
            "predictors": {
                key: block["prediction"] for key, block in result["results"].items()
            },
        }
        if "trust_assessment" in result:
            compact["trust_assessment"] = result["trust_assessment"]
        _print_json(compact)
        return

    print(f"\nView Finder comparison — {len(result['predictors'])} distinct predictors")
    for key in result["predictors"]:
        block = result["results"][key]
        print(f"\n{'=' * 66}")
        print(f"  {PREDICTOR_TITLES.get(key, key)}")
        print("=" * 66)
        warning = block.get("warning")
        if warning:
            print(f"! {warning}")
        gallery = block.get("gallery")
        if gallery:
            print(
                f"  gallery: v{gallery['version']} — {gallery['image_count']} reference photos"
                f" — matched {block['processing'].get('gallery_images_matched', 0)}"
                f" — closest single match "
                f"{block['processing'].get('top_similarity', float('nan')):.3f}"
            )
        _print_body(block)

    print(f"\n{'=' * 66}")
    trust = result.get("trust_assessment")
    if trust:
        decision = "EMIT" if trust["automation_accepted"] else "ABSTAIN"
        print(f"Trust decision: {decision} ({trust['trust_level']})")
        print(f"Reason:         {trust['reason']}")
        consensus = trust.get("consensus_prediction")
        if consensus:
            print(
                "Consensus:      "
                f"{consensus['latitude']:.6f}, {consensus['longitude']:.6f}"
            )
        print(
            "Important:      Agreement is not a guarantee; the gate requires "
            "held-out calibration before automated use."
        )
    else:
        print("Predictions above come from separate systems and are not combined.")
    print(f"JSON output: {output_path}")


def _print_single(result: dict[str, Any], output_path: Path, *, quiet: bool) -> None:
    """Print the human summary for a single-predictor run."""
    prediction = result.get("prediction")
    evidence = result.get("evidence_assessment")
    if quiet:
        compact: Any = prediction
        if evidence is not None:
            compact = {
                "prediction": prediction,
                "evidence_assessment": evidence,
            }
        _print_json(compact)
        return

    print("\nView Finder result")
    warning = result.get("warning")
    if warning:
        print(f"! {warning}")
    gallery = result.get("gallery")
    if gallery:
        print(
            f"Gallery:     v{gallery['version']} — {gallery['image_count']} reference photos"
            f" — matched {result['processing'].get('gallery_images_matched', 0)}"
            f" — closest single match "
            f"{result['processing'].get('top_similarity', float('nan')):.3f}"
        )
    if evidence is not None:
        decision = evidence["decision"].upper()
        print(f"Decision:    {decision} ({evidence['reliability_tier']})")
        print(f"Reason:      {evidence['reason']}")
    if prediction is None:
        hypotheses = result.get("visual_hypotheses") or []
        if hypotheses:
            best = hypotheses[0]
            print(
                "Hypothesis:  "
                f"{_place_label(best)} — {best['lat']:.6f}, {best['lon']:.6f} "
                "(manual review only)"
            )
        print(f"JSON output: {output_path}")
        return
    _print_body(result)
    print(f"JSON output: {output_path}")


def _print_body(block: dict[str, Any]) -> None:
    """Print one predictor's primary prediction and its alternatives."""
    prediction = block["prediction"]
    print(f"Location:    {_place_label(prediction)}")
    print(
        "Coordinates: "
        f"{prediction['latitude']:.6f}, {prediction['longitude']:.6f}"
    )
    evidence = block.get("evidence_assessment")
    if evidence is not None:
        radius = evidence.get("evidence_agreement_radius_km")
        radius_text = f"; channels agree within {radius:g} km" if radius is not None else ""
        print(f"Evidence:    {evidence['reliability_tier']}{radius_text}")
    elif "confidence_percent" in prediction:
        print(f"Confidence:  {prediction['confidence_percent']:.1f}%")
    else:
        print(f"Model score: {prediction['model_score']:.6g}")
    if "support_count" in prediction:
        print(
            f"Support:     {prediction['support_count']} frame(s), "
            f"{prediction['candidate_count']} candidate(s)"
        )

    alternatives = block.get("predictions", [])[1:]
    if alternatives:
        print("\nAlternative predictions")
        for alternative in alternatives:
            print(
                f"{alternative['rank']}. {_place_label(alternative)} — "
                f"{alternative['confidence']:.1f}% — "
                f"{alternative['lat']:.6f}, {alternative['lon']:.6f} — "
                f"support {alternative['support_count']}"
            )


def run(argv: list[str] | None = None) -> int:
    """Parse arguments, execute the selected pipeline, and return an exit code."""
    parser = build_parser()

    # --list-formats is informational and must work without --input, which is
    # otherwise required, so it is handled before argument parsing.
    if "--list-formats" in (argv if argv is not None else sys.argv[1:]):
        images = sorted(IMAGE_EXTENSIONS)
        videos = sorted(VIDEO_EXTENSIONS)
        print(f"Image formats ({len(images)}), decoded via Pillow:")
        print("  " + " ".join(images))
        print(f"\nVideo formats ({len(videos)}), decoded via OpenCV:")
        print("  " + " ".join(videos))
        print(
            "\nThe image list reflects the installed Pillow build; installing an "
            "optional plugin (e.g. pillow-heif for .heic) extends it automatically."
        )
        return 0

    args = parser.parse_args(argv)
    _resolve_predictor_default(args)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    progress_factory = _terminal_progress_factory(quiet=args.quiet, verbose=args.verbose)

    try:
        result, output_path = _execute(args, progress_factory=progress_factory)
    except UserInputError as exc:
        parser.exit(2, f"view-finder: error: {exc}\n")
    except PipelineError as exc:
        parser.exit(1, f"View Finder error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nview-finder: interrupted\n")
    except Exception as exc:  # pragma: no cover - last-resort guard
        # Nothing below the CLI should surface a raw traceback to a user. Log
        # the full detail under --verbose and show one actionable line.
        logger.debug("Unhandled error", exc_info=True)
        parser.exit(
            1,
            f"View Finder error: unexpected {exc.__class__.__name__}: {exc}\n"
            "Re-run with --verbose for the full traceback.\n",
        )

    _print_result(result, output_path, quiet=args.quiet)
    # The summary is the last thing printed at every verbosity. Under --quiet it
    # goes to stderr so stdout stays exactly the JSON that scripts parse.
    _print_summary(result, sys.stderr if args.quiet else sys.stdout)

    if (
        args.predictor == "trusted"
        and args.require_trust
        and not result["trust_assessment"]["automation_accepted"]
    ):
        return 3
    if (
        args.predictor in ("hybrid", "cloud-hybrid")
        and args.require_evidence
        and not result["evidence_assessment"]["automation_accepted"]
    ):
        return 3
    return 0


def _resolve_predictor_default(args: argparse.Namespace) -> None:
    """Fill in the default predictor and remember whether the user chose it.

    An explicit choice is honoured exactly, including failing when its
    prerequisites are missing. Only the *default* may quietly degrade.
    """
    args.predictor_is_explicit = args.predictor is not None
    if args.predictor is None:
        args.predictor = DEFAULT_PREDICTOR


def _execute(
    args: argparse.Namespace,
    *,
    progress_factory: ProgressFactory,
) -> tuple[dict[str, Any], Path]:
    """Run the pipeline for already-parsed arguments and return (result, path).

    Split out of :func:`run` so the desktop GUI can drive the identical
    orchestration with its own progress adapter, instead of duplicating it or
    shelling out to the CLI.
    """
    # Explicit flags win over the environment; leaving both unset keeps GeoCLIP's
    # bundled gallery, so default behaviour is unchanged.
    if args.gallery:
        os.environ["VIEWFINDER_GALLERY"] = str(Path(args.gallery).expanduser())
    if args.gallery_mode:
        os.environ["VIEWFINDER_GALLERY_MODE"] = args.gallery_mode

    source_metadata = _resolve_remote_source(args)
    input_path, input_type, output_path = _validate_args(args)
    # Predictor factories use args.input after validation. Normalize both the
    # positional and legacy forms to the resolved local path.
    args.input = str(input_path)
    args.source = None
    logger.debug("Input type: %s", input_type)
    logger.debug("Input path: %s", input_path)
    logger.debug("Output path: %s", output_path)

    specs = _predictor_specs(args)
    # Each predictor runs as a fully independent system over the same input.
    # Their raw outputs stay separate. Trusted mode may compute a geographic
    # consensus, but only an independently calibrated rule may emit it.
    outputs: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if input_type == "image":
            outputs[spec.key] = _process_image(
                input_path,
                args.top_k,
                args.max_predictions,
                spec,
                remove_text_enabled=args.remove_text,
                crop_screen=args.crop_screen,
            )
        else:
            outputs[spec.key] = _process_video(
                input_path,
                args.fps,
                args.max_frames,
                args.max_video_seconds,
                args.top_k,
                args.max_predictions,
                progress_factory,
                spec,
            )

    result: dict[str, Any] = {
        "input": str(input_path),
        "input_type": input_type,
        "source": source_metadata
        or {
            "kind": "local_file",
            "path": str(input_path),
        },
    }
    if len(specs) == 1:
        # Single-predictor runs keep the original flat shape, so existing
        # consumers of the JSON are unaffected.
        result.update(outputs[specs[0].key])
    else:
        result["predictors"] = [spec.key for spec in specs]
        result["results"] = outputs
        if args.predictor == "trusted":
            assess_trust = _stage(
                "Loading trust assessment support",
                lambda: _import_attr("trust", "assess_trust"),
            )
            assessment = _stage(
                "Checking cross-model agreement and held-out calibration",
                lambda: assess_trust(
                    outputs,
                    agreement_radius_km=args.trust_radius_km,
                    required_success_rate=args.required_success_rate,
                    success_definition_km=args.trust_success_km,
                    calibration_path=args.trust_calibration,
                ),
            )
            consensus = assessment.get("consensus_prediction")
            if consensus:
                reverse_geocode = _stage(
                    "Loading offline place index for consensus",
                    lambda: _import_attr("reverse_geocode", "reverse_geocode"),
                    hint="Install reverse_geocoder from viewfinder/requirements.txt.",
                )
                place = _stage(
                    "Finding nearest named place for consensus",
                    lambda: reverse_geocode(
                        consensus["latitude"],
                        consensus["longitude"],
                    ),
                )
                consensus.update(place)
            result["trust_assessment"] = assessment
            result["prediction"] = (
                consensus if assessment["automation_accepted"] else None
            )
    _stage(
        "Writing JSON result",
        lambda: _write_result(output_path, result),
        hint="Check that the output directory exists and is writable.",
    )
    return result, output_path


def analyze(
    source: str,
    *,
    predictor: str | None = None,
    output: str | None = None,
    extra_args: list[str] | None = None,
    progress_factory: ProgressFactory | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run the pipeline programmatically and return ``(result, output_path)``.

    The public seam for non-CLI callers, principally :mod:`viewfinder.gui`.
    Arguments are routed through the real parser rather than a parallel code
    path, so defaults and validation behave identically to the command line.

    Raises:
        UserInputError: An invalid argument or unusable input file.
        PipelineError: A stage failed; the message is user-presentable.
    """
    argv: list[str] = [source]
    if predictor:
        argv += ["--predictor", predictor]
    if output:
        argv += ["--output", output]
    argv += extra_args or []

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits on malformed input; translate it
        raise UserInputError(f"invalid arguments: {' '.join(argv)}") from exc
    _resolve_predictor_default(args)

    def null_factory(label: str) -> TerminalProgressBar:
        return TerminalProgressBar(label, enabled=False)

    return _execute(args, progress_factory=progress_factory or null_factory)


def main() -> None:
    """Console-script compatible entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
