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
from typing import Any, Callable, TypeVar


logger = logging.getLogger("viewfinder")

# Accepted formats live in one module so the CLI can never drift from what the
# image loader supports. Imported both ways because main.py runs as a module
# (`python -m viewfinder.main`) and as a script (`python viewfinder/main.py`).
try:
    from .formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
except ImportError:  # pragma: no cover - script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from viewfinder.formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
DEFAULT_FPS = 1.0
DEFAULT_TOP_K = 5
DEFAULT_MAX_PREDICTIONS = 5

_T = TypeVar("_T")


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
            "hybrid",
            "cloud-hybrid",
            "geoclip",
            "retrieval",
            "both",
            "ensemble",
            "plonk-osv",
            "plonk-yfcc",
            "trusted",
        ),
        default="hybrid",
        help=(
            "which geolocation system to run: hybrid (default) uses fully local "
            "EXIF/OCR/GeoNames/Commons evidence; cloud-hybrid is the optional Google "
            "implementation; geoclip and PLONK are visual models; retrieval matches "
            "a reference gallery; both compares GeoCLIP and retrieval; ensemble merges "
            "them, letting retrieval override GeoCLIP only where the gallery has a "
            "strong reference match; trusted runs GeoCLIP plus both PLONK models "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--ensemble-trust-threshold",
        type=float,
        default=None,
        help=(
            "with --predictor ensemble, the retrieval top-1 cosine similarity at or "
            "above which retrieval's answer becomes primary (default: calibrated 0.825)"
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
        "--gallery-path",
        metavar="PATH",
        help=(
            "gallery version directory built by gallery-pipeline's load stage, e.g. "
            "gallery-pipeline/data/target-regions/gallery/v1 "
            "(required for --predictor retrieval or both)"
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

    if args.ensemble_trust_threshold is not None:
        if args.predictor != "ensemble":
            raise UserInputError(
                "--ensemble-trust-threshold is only valid with --predictor ensemble"
            )
        if not 0.0 <= args.ensemble_trust_threshold <= 1.0:
            raise UserInputError(
                "--ensemble-trust-threshold must be a cosine similarity in [0, 1]"
            )

    if args.predictor in ("retrieval", "both", "ensemble"):
        if not args.gallery_path:
            raise UserInputError(
                f"--predictor {args.predictor} requires --gallery-path pointing at a gallery "
                "version directory built by gallery-pipeline, "
                "e.g. --gallery-path gallery-pipeline/data/target-regions/gallery/v1"
            )
        gallery_path = Path(args.gallery_path).expanduser().resolve()
        if not gallery_path.is_dir():
            raise UserInputError(f"--gallery-path is not a directory: {gallery_path}")

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


def _new_geolocator() -> Any:
    """Construct GeoCLIP in enforced offline mode.

    Any custom gallery selection is read from the environment, which the CLI
    flags populate in :func:`run`. This keeps the construction signature stable
    for both the image and video pipelines.
    """
    geolocator_class = _import_attr("geolocator", "GeoLocator")
    return geolocator_class(offline=True)


def _new_retrieval_geolocator(gallery_path: str) -> Any:
    """Construct the image-to-image retrieval predictor for one gallery."""
    retrieval_class = _import_attr("retrieval_geolocator", "RetrievalGeolocator")
    return retrieval_class(gallery_path)


def _new_ensemble_geolocator(gallery_path: str, trust_threshold: float | None):
    """Compose GeoCLIP and retrieval behind the calibrated trust threshold.

    Both underlying predictors are built exactly as their own --predictor modes
    build them; the ensemble only decides which answer leads.
    """
    ensemble_class = _import_attr("ensemble_geolocator", "EnsembleGeolocator")
    default_threshold = _import_attr("ensemble_geolocator", "DEFAULT_TRUST_THRESHOLD")
    return ensemble_class(
        geoclip=_new_geolocator(),
        retrieval=_new_retrieval_geolocator(gallery_path),
        trust_threshold=default_threshold if trust_threshold is None else trust_threshold,
    )


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
    # Enables the retrieval-only weak-similarity re-rank and the best-match
    # caution in the CLI. None for GeoCLIP, whose scores have no similarity scale.
    weak_similarity_threshold: float | None = None


def _predictor_specs(args: argparse.Namespace) -> list[PredictorSpec]:
    """Return the predictors selected by --predictor, in report order."""
    weak_threshold = _import_attr("aggregator", "DEFAULT_WEAK_SIMILARITY_THRESHOLD")
    geoclip = PredictorSpec(
        key="geoclip",
        label="GeoCLIP",
        build=_new_geolocator,
        load_hint=(
            "Install the requirements and pre-cache the GeoCLIP weights; "
            "the CLI does not access the network."
        ),
    )
    gallery_path = str(Path(args.gallery_path).expanduser()) if args.gallery_path else ""
    retrieval = PredictorSpec(
        key="retrieval",
        label="Retrieval",
        build=lambda: _new_retrieval_geolocator(gallery_path),
        load_hint=(
            "Confirm --gallery-path points at a gallery version directory containing "
            "gallery.faiss, metadata.parquet and manifest.json, and that faiss-cpu, "
            "open-clip-torch and pyarrow are installed."
        ),
        weak_similarity_threshold=weak_threshold,
    )
    ensemble = PredictorSpec(
        key="ensemble",
        label="Ensemble (GeoCLIP + Retrieval)",
        build=lambda: _new_ensemble_geolocator(gallery_path, args.ensemble_trust_threshold),
        load_hint=(
            "The ensemble needs both predictors: cached GeoCLIP weights and a "
            "--gallery-path with gallery.faiss, metadata.parquet and manifest.json."
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
    if args.predictor == "retrieval":
        return [retrieval]
    if args.predictor == "ensemble":
        return [ensemble]
    if args.predictor == "plonk-osv":
        return [plonk_osv]
    if args.predictor == "plonk-yfcc":
        return [plonk_yfcc]
    if args.predictor == "trusted":
        return [geoclip, plonk_osv, plonk_yfcc]
    return [geoclip, retrieval]


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
        hint="Confirm the file is a readable image in a supported format (--list-formats).",
    )
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
            weak_similarity_threshold=spec.weak_similarity_threshold,
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
    if spec.weak_similarity_threshold is not None:
        output["weak_similarity_threshold"] = spec.weak_similarity_threshold
    output["processing"] = {
        "candidates_returned": len(candidates),
        "clusters_returned": len(aggregate_result.predictions),
    }
    if text_removal_info is not None:
        output["processing"]["text_removal"] = text_removal_info
    _attach_gallery_details(output, geolocator, candidates)
    _attach_model_provenance(output, geolocator)
    _attach_evidence_assessment(output, geolocator)
    return output


def _attach_gallery_details(
    output: dict[str, Any],
    geolocator: Any,
    candidates: Any,
) -> None:
    """Record retrieval-specific provenance, when the predictor has any.

    GeoCLIP has no reference gallery, so this is a no-op for it.
    """
    info = getattr(geolocator, "info", None)
    if info is None or not hasattr(info, "as_dict"):
        return
    output["gallery"] = info.as_dict()
    warning = info.warning()
    if warning:
        output["warning"] = warning

    matched = _import_attr("retrieval_geolocator", "matched_image_ids")
    if isinstance(candidates, list) and candidates and not isinstance(candidates[0], list):
        pool = candidates
    else:
        pool = [candidate for frame in candidates for candidate in frame.candidates]
    output["processing"]["gallery_images_matched"] = len(matched(pool))
    output["processing"]["gallery_size"] = info.image_count

    # Raw cosine similarity of the single best match. This is deliberately
    # reported alongside the confidence score because the two answer different
    # questions: confidence measures how well the returned neighbours agree with
    # each other, while this measures how much the query actually resembles the
    # closest reference photo. A tight cluster of poor matches yields high
    # confidence and a low top similarity, which is the signature of a query
    # whose true location is simply absent from the gallery.
    if pool:
        output["processing"]["top_similarity"] = round(max(c.score for c in pool), 4)


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
    quiet: bool,
    spec: PredictorSpec,
) -> dict[str, Any]:
    """Run frame extraction, batched inference, aggregation, and place lookup."""
    extract_frames = _stage(
        "Loading video support",
        lambda: _import_attr("video_processor", "extract_frames"),
        hint="Install OpenCV and the other packages in viewfinder/requirements.txt.",
    )
    frames, stats = _stage(
        f"Sampling video at {fps:g} fps and filtering blurry frames",
        lambda: extract_frames(str(input_path), fps=fps, max_frames=max_frames),
        hint="Confirm that OpenCV can decode this video and that the file is not corrupt.",
    )
    if not frames:
        raise PipelineError(
            "Video frame extraction produced zero usable frames after blur filtering. "
            "Try a clearer clip or a different sampling rate with --fps."
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
    predictions = _stage(
        f"Processing {len(frames)} video frames with {spec.label}",
        lambda: geolocator.predict_frames(frames, top_k=top_k, progress=not quiet),
        hint="Try a lower --fps or verify that the selected Torch device has enough memory.",
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
            weak_similarity_threshold=spec.weak_similarity_threshold,
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
    if spec.weak_similarity_threshold is not None:
        output["weak_similarity_threshold"] = spec.weak_similarity_threshold
    output["processing"] = processing
    _attach_gallery_details(output, geolocator, predictions)
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
    "retrieval": "Retrieval (nearest match among our own scraped reference photos)",
    "plonk-osv": "PLONK OSV-5M (probabilistic street-view model)",
    "plonk-yfcc": "PLONK YFCC (probabilistic general-photo model)",
}


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


def _best_match_note(block: dict[str, Any]) -> str:
    """Annotate the confidence line with the retrieval primary's best match.

    Retrieval confidence measures how well the returned neighbours agree, not
    how good the match is. A high confidence built on a low best-match cosine
    similarity is the tell that the query's true location may be absent from the
    gallery, so that case is flagged inline on the primary result rather than
    left buried in JSON. GeoCLIP has no similarity scale, so this is a no-op for
    it (the threshold key is only written for retrieval).
    """
    threshold = block.get("weak_similarity_threshold")
    if threshold is None:
        return ""
    top_similarity = block.get("prediction", {}).get("top_similarity")
    if top_similarity is None:
        return ""
    if top_similarity < threshold:
        return f" (best match: {top_similarity:.2f} — weak, treat with caution)"
    return f" (best match: {top_similarity:.2f})"


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
        print(f"Confidence:  {prediction['confidence_percent']:.1f}%{_best_match_note(block)}")
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
    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    # Explicit flags win over the environment; leaving both unset keeps GeoCLIP's
    # bundled gallery, so default behaviour is unchanged.
    if args.gallery:
        os.environ["VIEWFINDER_GALLERY"] = str(Path(args.gallery).expanduser())
    if args.gallery_mode:
        os.environ["VIEWFINDER_GALLERY_MODE"] = args.gallery_mode

    try:
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
                )
            else:
                outputs[spec.key] = _process_video(
                    input_path,
                    args.fps,
                    args.max_frames,
                    args.max_video_seconds,
                    args.top_k,
                    args.max_predictions,
                    args.quiet,
                    spec,
                )

        result = {
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
    except UserInputError as exc:
        parser.exit(2, f"view-finder: error: {exc}\n")
    except PipelineError as exc:
        parser.exit(1, f"View Finder error: {exc}\n")

    _print_result(result, output_path, quiet=args.quiet)
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


def main() -> None:
    """Console-script compatible entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
