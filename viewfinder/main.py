"""Command-line entry point for the View Finder geolocation pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar


logger = logging.getLogger("viewfinder")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
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
            "using offline GeoCLIP inference."
        ),
    )
    parser.add_argument("--input", required=True, metavar="PATH", help="image or video file")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        metavar="FLOAT",
        help="video sampling rate in frames per second (default: %(default)s)",
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
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", "-v", action="store_true", help="enable debug logging for every stage"
    )
    verbosity.add_argument(
        "--quiet", "-q", action="store_true", help="suppress progress; print only final result"
    )
    return parser


def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    """Configure CLI logging without changing the pipeline modules."""
    level = logging.CRITICAL if quiet else logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)


def _validate_args(args: argparse.Namespace) -> tuple[Path, str, Path]:
    """Validate cheap input facts before importing or initializing ML code."""
    input_path = Path(args.input).expanduser().resolve()
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
        supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
        raise UserInputError(
            f"unsupported input format '{extension or '(none)'}'; supported formats: {supported}"
        )

    if input_type == "video" and (not math.isfinite(args.fps) or args.fps <= 0):
        raise UserInputError("--fps must be a finite number greater than zero")
    if args.top_k <= 0:
        raise UserInputError("--top-k must be greater than zero")
    if args.max_predictions <= 0:
        raise UserInputError("--max-predictions must be greater than zero")

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
    """Construct GeoCLIP in enforced offline mode."""
    geolocator_class = _import_attr("geolocator", "GeoLocator")
    return geolocator_class(offline=True)


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
        }
    )
    return {
        "prediction": primary,
        "predictions": ranked,
        "alternatives": ranked[1:],
    }


def _process_image(input_path: Path, top_k: int, max_predictions: int) -> dict[str, Any]:
    """Run image retrieval, multi-hypothesis clustering, and place lookup."""
    load_image = _stage(
        "Loading image support",
        lambda: _import_attr("video_processor", "load_image"),
        hint="Install dependencies with 'python -m pip install -r viewfinder/requirements.txt'.",
    )
    image = _stage(
        "Loading image",
        lambda: load_image(str(input_path)),
        hint="Confirm that the file is a valid, readable JPEG or PNG.",
    )
    geolocator = _stage(
        "Loading GeoCLIP model",
        _new_geolocator,
        hint=(
            "Install the requirements and pre-cache the GeoCLIP weights; "
            "the CLI does not access the network."
        ),
    )
    candidates = _stage(
        "Analyzing image",
        lambda: geolocator.predict_image(image, top_k=top_k),
        hint="Try a smaller image or verify that the selected Torch device has enough memory.",
    )
    if not candidates:
        raise PipelineError("Analyzing image failed: GeoCLIP returned no location candidates.")

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
    output["processing"] = {
        "candidates_returned": len(candidates),
        "clusters_returned": len(aggregate_result.predictions),
    }
    return output


def _process_video(
    input_path: Path,
    fps: float,
    top_k: int,
    max_predictions: int,
    quiet: bool,
) -> dict[str, Any]:
    """Run frame extraction, batched inference, aggregation, and place lookup."""
    extract_frames = _stage(
        "Loading video support",
        lambda: _import_attr("video_processor", "extract_frames"),
        hint="Install OpenCV and the other packages in viewfinder/requirements.txt.",
    )
    frames, stats = _stage(
        f"Sampling video at {fps:g} fps and filtering blurry frames",
        lambda: extract_frames(str(input_path), fps=fps),
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
        "Loading GeoCLIP model",
        _new_geolocator,
        hint=(
            "Install the requirements and pre-cache the GeoCLIP weights; "
            "the CLI does not access the network."
        ),
    )
    predictions = _stage(
        f"Processing {len(frames)} video frames",
        lambda: geolocator.predict_frames(frames, top_k=top_k, progress=not quiet),
        hint="Try a lower --fps or verify that the selected Torch device has enough memory.",
    )
    if not predictions:
        raise PipelineError("GeoCLIP inference produced no frame predictions.")
    if any(not prediction.candidates for prediction in predictions):
        raise PipelineError("GeoCLIP inference returned an empty candidate list for a frame.")

    aggregate = _stage(
        "Loading aggregation support",
        lambda: _import_attr("aggregator", "aggregate"),
    )
    aggregate_result = _stage(
        "Clustering frame predictions",
        lambda: aggregate(
            predictions,
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
    output = _serialize_aggregate(aggregate_result, places)
    output["processing"] = processing
    return output


def _write_result(output_path: Path, result: dict[str, Any]) -> None:
    """Create the output directory if needed and write formatted UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _place_label(prediction: dict[str, Any]) -> str:
    """Format non-empty reverse-geocoded place components."""
    parts = [prediction.get("city"), prediction.get("admin1"), prediction.get("country")]
    return ", ".join(str(part) for part in parts if part) or "nearest known place unavailable"


def _print_result(result: dict[str, Any], output_path: Path, *, quiet: bool) -> None:
    """Print either a compact machine-readable result or a human summary."""
    prediction = result["prediction"]
    if quiet:
        print(json.dumps(prediction, ensure_ascii=False, allow_nan=False))
        return

    print("\nView Finder result")
    print(f"Location:    {_place_label(prediction)}")
    print(
        "Coordinates: "
        f"{prediction['latitude']:.6f}, {prediction['longitude']:.6f}"
    )
    if "confidence_percent" in prediction:
        print(f"Confidence:  {prediction['confidence_percent']:.1f}%")
    else:
        print(f"Model score: {prediction['model_score']:.6g}")
    if "support_count" in prediction:
        print(
            f"Support:     {prediction['support_count']} frame(s), "
            f"{prediction['candidate_count']} candidate(s)"
        )

    alternatives = result.get("predictions", [])[1:]
    if alternatives:
        print("\nAlternative predictions")
        for alternative in alternatives:
            print(
                f"{alternative['rank']}. {_place_label(alternative)} — "
                f"{alternative['confidence']:.1f}% — "
                f"{alternative['lat']:.6f}, {alternative['lon']:.6f} — "
                f"support {alternative['support_count']}"
            )
    print(f"JSON output: {output_path}")


def run(argv: list[str] | None = None) -> int:
    """Parse arguments, execute the selected pipeline, and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        input_path, input_type, output_path = _validate_args(args)
        logger.debug("Input type: %s", input_type)
        logger.debug("Input path: %s", input_path)
        logger.debug("Output path: %s", output_path)

        if input_type == "image":
            pipeline_result = _process_image(
                input_path,
                args.top_k,
                args.max_predictions,
            )
        else:
            pipeline_result = _process_video(
                input_path,
                args.fps,
                args.top_k,
                args.max_predictions,
                args.quiet,
            )

        result = {
            "input": str(input_path),
            "input_type": input_type,
            **pipeline_result,
        }
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
    return 0


def main() -> None:
    """Console-script compatible entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
