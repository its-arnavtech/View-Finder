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
    parser.add_argument(
        "--predictor",
        choices=("geoclip", "retrieval", "both"),
        default="geoclip",
        help=(
            "which geolocation system to run: geoclip (default) uses learned location "
            "embeddings; retrieval matches against our own scraped reference photos; "
            "both runs each independently and reports them side by side (default: %(default)s)"
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

    if args.predictor in ("retrieval", "both"):
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


@dataclass(frozen=True)
class PredictorSpec:
    """How to build and label one predictor.

    The two predictors are independent systems that happen to share the
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

    if args.predictor == "geoclip":
        return [geoclip]
    if args.predictor == "retrieval":
        return [retrieval]
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
        hint="Confirm that the file is a valid, readable JPEG or PNG.",
    )
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
    _attach_gallery_details(output, geolocator, candidates)
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


def _process_video(
    input_path: Path,
    fps: float,
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
    output = _serialize_aggregate(aggregate_result, places)
    output["predictor"] = spec.key
    if spec.weak_similarity_threshold is not None:
        output["weak_similarity_threshold"] = spec.weak_similarity_threshold
    output["processing"] = processing
    _attach_gallery_details(output, geolocator, predictions)
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


PREDICTOR_TITLES = {
    "geoclip": "GeoCLIP (learned location embeddings; no reference photos)",
    "retrieval": "Retrieval (nearest match among our own scraped reference photos)",
}


def _print_result(result: dict[str, Any], output_path: Path, *, quiet: bool) -> None:
    """Print either a compact machine-readable result or a human summary."""
    if "results" in result:
        _print_comparison(result, output_path, quiet=quiet)
        return
    _print_single(result, output_path, quiet=quiet)


def _print_comparison(result: dict[str, Any], output_path: Path, *, quiet: bool) -> None:
    """Print each predictor's result in its own labelled block.

    The two systems are reported separately and never reconciled: where they
    disagree, that disagreement is the useful signal.
    """
    if quiet:
        print(
            json.dumps(
                {key: block["prediction"] for key, block in result["results"].items()},
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return

    print("\nView Finder comparison — two independent predictors")
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
    print("Predictions above come from separate systems and are not combined.")
    print(f"JSON output: {output_path}")


def _print_single(result: dict[str, Any], output_path: Path, *, quiet: bool) -> None:
    """Print the human summary for a single-predictor run."""
    prediction = result["prediction"]
    if quiet:
        print(json.dumps(prediction, ensure_ascii=False, allow_nan=False))
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
    if "confidence_percent" in prediction:
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
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    # Explicit flags win over the environment; leaving both unset keeps GeoCLIP's
    # bundled gallery, so default behaviour is unchanged.
    if args.gallery:
        os.environ["VIEWFINDER_GALLERY"] = str(Path(args.gallery).expanduser())
    if args.gallery_mode:
        os.environ["VIEWFINDER_GALLERY_MODE"] = args.gallery_mode

    try:
        input_path, input_type, output_path = _validate_args(args)
        logger.debug("Input type: %s", input_type)
        logger.debug("Input path: %s", input_path)
        logger.debug("Output path: %s", output_path)

        specs = _predictor_specs(args)
        # Each predictor runs as a fully independent system over the same input.
        # Their outputs are kept in separate blocks and never merged or averaged:
        # they use different models and different evidence, so a combined number
        # would not mean anything.
        outputs: dict[str, dict[str, Any]] = {}
        for spec in specs:
            if input_type == "image":
                outputs[spec.key] = _process_image(
                    input_path,
                    args.top_k,
                    args.max_predictions,
                    spec,
                )
            else:
                outputs[spec.key] = _process_video(
                    input_path,
                    args.fps,
                    args.top_k,
                    args.max_predictions,
                    args.quiet,
                    spec,
                )

        result = {
            "input": str(input_path),
            "input_type": input_type,
        }
        if len(specs) == 1:
            # Single-predictor runs keep the original flat shape, so existing
            # consumers of the JSON are unaffected.
            result.update(outputs[specs[0].key])
        else:
            result["predictors"] = [spec.key for spec in specs]
            result["results"] = outputs
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
