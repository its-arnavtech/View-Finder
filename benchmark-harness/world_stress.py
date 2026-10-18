"""Stress the World Engine against many real photos and awkward inputs.

The engine's contract is "any still image in, a usable 3D world out". This
runs it over a seeded random sample of the repository's travel photos and
street-level benchmark images, plus generated edge cases (extreme aspect
ratios, tiny/huge, every odd pixel mode, all-sky, black, noise, EXIF
rotation, animated GIF...). Every world must pass
``viewfinder.world.validate.validate_world``.

    python benchmark-harness/world_stress.py --photos 120 --street 40 --quality mobile

Writes ``results.csv`` and ``summary.json`` under ``--out`` and exits 1 if any
input that should have produced a world did not.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from viewfinder.world import build_world  # noqa: E402
from viewfinder.world.depth import load_estimator  # noqa: E402
from viewfinder.world.ingest import ImageInputError  # noqa: E402
from viewfinder.world.validate import validate_world  # noqa: E402


def _scene(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A plausible outdoor-ish picture: sky gradient, ground, a few blocks."""
    rng = np.random.default_rng(seed)
    rows = np.linspace(0, 1, height)[:, None, None]
    sky = np.array([70, 120, 210]) * (1 - rows) + np.array([200, 220, 240]) * rows
    ground = np.array([90, 110, 60]) * (1 - rows) + np.array([50, 70, 30]) * rows
    horizon = int(height * 0.45)
    image = np.where(np.arange(height)[:, None, None] < horizon, sky, ground)
    image = np.broadcast_to(image, (height, width, 3)).copy()
    for _ in range(6):
        x = rng.integers(0, max(1, width - width // 6))
        w = rng.integers(max(2, width // 20), max(3, width // 6))
        top = rng.integers(horizon // 3, max(horizon // 3 + 1, horizon))
        image[top : horizon + (height - horizon) // 4, x : x + w] = rng.integers(40, 200, 3)
    image += rng.normal(0, 6, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def edge_cases(directory: Path, sky_source: Path | None) -> list[tuple[Path, bool]]:
    """Generate awkward inputs. Returns (path, should_build) pairs."""
    directory.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[Path, bool]] = []

    def save(name: str, image: Image.Image, should_build: bool = True, **kwargs) -> None:
        path = directory / name
        image.save(path, **kwargs)
        cases.append((path, should_build))

    save("panorama_8x1.jpg", Image.fromarray(_scene(4000, 500, 1)), quality=88)
    save("tall_1x6.jpg", Image.fromarray(_scene(300, 1800, 2)), quality=88)
    save("tiny_48x36.png", Image.fromarray(_scene(48, 36, 3)))
    save("huge_7200x4800.jpg", Image.fromarray(_scene(7200, 4800, 4)), quality=70)
    save("grayscale_L.jpg", Image.fromarray(_scene(900, 600, 5)).convert("L"), quality=90)
    save("cmyk.jpg", Image.fromarray(_scene(900, 600, 6)).convert("CMYK"), quality=90)
    rgba = Image.fromarray(_scene(800, 600, 7)).convert("RGBA")
    alpha = np.full((600, 800), 255, np.uint8)
    alpha[:, :400] = 0
    rgba.putalpha(Image.fromarray(alpha))
    save("half_transparent.png", rgba)
    gray16 = (np.asarray(Image.fromarray(_scene(640, 480, 8)).convert("L"), np.uint16) * 257)
    save("gray16.png", Image.fromarray(gray16.astype(np.uint16)))
    frames = [Image.fromarray(_scene(400, 300, s)).convert("P") for s in (9, 10, 11)]
    path = directory / "animated.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    cases.append((path, True))
    save("uniform_gray.png", Image.new("RGB", (800, 600), (128, 128, 128)))
    save("all_black.jpg", Image.new("RGB", (1200, 800), (0, 0, 0)), quality=90)
    noise = np.random.default_rng(12).integers(0, 256, (600, 800, 3), dtype=np.uint8)
    save("white_noise.png", Image.fromarray(noise))
    exif = Image.Exif()
    exif[0x0112] = 8
    save("exif_rotate_270.jpg", Image.fromarray(_scene(1200, 800, 13)), quality=88, exif=exif)
    save("lzw.tif", Image.fromarray(_scene(800, 600, 14)), compression="tiff_lzw")
    save("photo.webp", Image.fromarray(_scene(800, 600, 15)), quality=85)
    save("photo.bmp", Image.fromarray(_scene(640, 480, 16)))
    if sky_source is not None:
        with Image.open(sky_source) as source:
            source = source.convert("RGB")
            sky_only = source.crop((0, 0, source.width, max(40, source.height // 5)))
        save("real_sky_only.jpg", sky_only, quality=92)
    save("too_small_12x12.png", Image.new("RGB", (12, 12)), should_build=False)
    garbage = directory / "not_an_image.jpg"
    garbage.write_bytes(b"\xff\xd8\xff\xe0" + b"garbage" * 50)
    cases.append((garbage, False))
    return cases


def real_photos(photos: int, street: int, seed: int) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    travel = sorted(p for p in (REPO / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    chosen = [(p, "travel") for p in rng.sample(travel, min(photos, len(travel)))]
    manifest = REPO / "benchmark-data" / "manifest.csv"
    if street and manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = [Path(r["image_path"]) for r in csv.DictReader(handle)]
        rows = [p for p in rows if p.is_file()]
        chosen += [(p, "street") for p in rng.sample(rows, min(street, len(rows)))]
    return chosen


def run_one(path: Path, kind: str, should_build: bool, quality: str, estimator, inpainter=None) -> dict:
    row: dict = {"name": path.name, "kind": kind, "expected": "world" if should_build else "rejection"}
    try:
        with Image.open(path) as probe:
            row["size"] = f"{probe.width}x{probe.height}"
            row["mode"] = probe.mode
    except Exception:
        row["size"], row["mode"] = "?", "?"
    started = time.perf_counter()
    try:
        result = build_world(path, profile=quality, estimator=estimator, inpainter=inpainter)
    except ImageInputError as exc:
        row.update(outcome="rejected", detail=str(exc), ok=not should_build)
        return row
    except Exception as exc:  # the finding we are looking for
        row.update(outcome="crash", detail=f"{type(exc).__name__}: {exc}", ok=False,
                   trace=traceback.format_exc(limit=3))
        return row
    problems = validate_world(result)
    manifest = result.manifest
    layers = manifest["layers"]
    row.update(
        outcome="world" if not problems else "invalid",
        ok=should_build and not problems,
        detail="; ".join(problems),
        ms=round((time.perf_counter() - started) * 1000),
        mb=round(len(result.glb) / 1e6, 3),
        triangles=sum(layer["triangles"] for layer in layers.values()),
        coverage=manifest["quality"]["foreground_coverage"],
        sky=manifest["environment"]["sky_fraction"],
        ground=manifest["environment"]["ground"]["accepted"],
        layers="+".join(layers),
        warnings=" | ".join(manifest["warnings"]),
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", type=int, default=120, help="random travel photos from images/")
    parser.add_argument("--street", type=int, default=40, help="random street-level benchmark images")
    parser.add_argument("--quality", default="mobile")
    parser.add_argument("--depth-model", default="depth-anything-v2-small")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=REPO / "local-data" / "world-stress")
    parser.add_argument("--skip-edge", action="store_true")
    parser.add_argument("--inpaint", choices=("lama", "none"), default="none",
                        help="also run LaMa inpainting (slow on CPU: ~2-10 s per photo)")
    args = parser.parse_args(argv)

    estimator = load_estimator(args.depth_model)
    inpainter = None
    if args.inpaint == "lama":
        from viewfinder.world.inpaint import load_inpainter

        inpainter = load_inpainter()
    inputs: list[tuple[Path, str, bool]] = [(p, k, True) for p, k in real_photos(args.photos, args.street, args.seed)]
    if not args.skip_edge:
        sky_source = REPO / "images" / "Alaska_00001_30657829_8217c3e79d_22_58542460@N00.jpg"
        inputs += [(p, "edge", s) for p, s in edge_cases(args.out / "edge-cases", sky_source if sky_source.is_file() else None)]

    rows = []
    for index, (path, kind, should_build) in enumerate(inputs, 1):
        row = run_one(path, kind, should_build, args.quality, estimator, inpainter)
        rows.append(row)
        flag = "ok  " if row["ok"] else "FAIL"
        print(f"[{index:3}/{len(inputs)}] {flag} {kind:6} {row['outcome']:8} {path.name[:48]:48} "
              f"{row.get('ms', ''):>6} {row.get('detail', '')[:80]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key != "trace"})
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    worlds = [r for r in rows if r["outcome"] == "world"]
    failures = [r for r in rows if not r["ok"]]

    def stats(key: str) -> dict:
        values = [r[key] for r in worlds if isinstance(r.get(key), (int, float))]
        if not values:
            return {}
        values.sort()
        return {"median": statistics.median(values), "p95": values[int(0.95 * (len(values) - 1))],
                "min": values[0], "max": values[-1]}

    summary = {
        "quality": args.quality,
        "inputs": len(rows),
        "by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in ("travel", "street", "edge")},
        "worlds": len(worlds),
        "expected_rejections": sum(1 for r in rows if r["outcome"] == "rejected" and r["ok"]),
        "failures": [{k: r.get(k) for k in ("name", "kind", "outcome", "detail", "trace")} for r in failures],
        "build_ms": stats("ms"),
        "glb_mb": stats("mb"),
        "triangles": stats("triangles"),
        "foreground_coverage": stats("coverage"),
        "sky_fraction": stats("sky"),
        "ground_accepted_rate": round(sum(1 for r in worlds if r["ground"]) / max(1, len(worlds)), 3),
        "low_coverage_worlds": [r["name"] for r in worlds if r["coverage"] < 0.6],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    print(f"\n{len(failures)} failure(s)")
    for failure in failures:
        print(f"  {failure['name']}: {failure['outcome']} {failure.get('detail', '')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
