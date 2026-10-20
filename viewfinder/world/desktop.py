"""The World Engine as the desktop app's child process: JSON lines on stdio.

The Electron shell (``desktop/``) starts ``view-finder world desktop`` and
talks to it over stdin/stdout, so the app opens no network port at all. One
request per line, one reply per line, matched by ``id``::

    -> {"id": 1, "op": "status"}
    <- {"id": 1, "ok": true, "result": {"state": "ready", ...}}
    -> {"id": 2, "op": "build", "path": "/…/uploads/3f….photo", "quality": "desktop", "name": "lake"}
    <- {"id": 2, "ok": true, "result": {"job_id": "…"}}
    -> {"id": 3, "op": "job", "job_id": "…"}
    <- {"id": 3, "ok": true, "result": {"status": "running", "stage": "depth", "progress": 0.31}}

Builds reuse the server's ``JobRunner`` and ``WorldStore``: one worker
thread, worlds kept under ``--store`` (the app reads them from there). The
models load in the background, so ``status`` answers at once with
``loading``, then ``ready`` or ``failed``. End of stdin (the app quit or
crashed) ends the process.

Stdout carries the protocol only. Libraries that print (progress bars,
native code) would corrupt it, so file descriptor 1 is pointed at stderr
and the protocol keeps a private duplicate of the original stdout.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from .profiles import PROFILES
from .server import ID_PATTERN, JobRunner, QueueFull, WorldStore, _looks_like_image

logger = logging.getLogger(__name__)

PROTOCOL = 1


@dataclass
class DesktopConfig:
    store: Path
    max_worlds: int = 30
    max_upload_mb: int = 25
    gpu: str = "nvidia"  # nvidia: every GPU stage on the NVIDIA card, or fail; any: the system's choice


# (estimator, inpainter, generator, classify)
Engine = tuple[Any, Any, Any, Any]


def gpu_report(policy: str) -> dict[str, Any]:
    """What the engine's GPU stages will run on; ``problems`` is empty when
    the policy holds."""
    report: dict[str, Any] = {"policy": policy, "cuda": None, "renderer": None, "problems": []}
    try:
        import torch

        if torch.cuda.is_available():
            report["cuda"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # a broken torch install is reported, not fatal here
        report["problems"].append(f"PyTorch could not be loaded ({exc.__class__.__name__}).")
    try:
        from .render import RendererUnavailable, WorldRenderer, WrongGPU

        try:
            renderer = WorldRenderer()
            report["renderer"] = renderer.gpu
            renderer.release()
        except WrongGPU as exc:
            report["problems"].append(str(exc))
        except RendererUnavailable:
            pass  # only generated worlds walk in; without moderngl a world is seen from one point
    except ImportError:
        pass
    if policy == "nvidia" and report["cuda"] is None:
        report["problems"].append(
            "PyTorch cannot see an NVIDIA GPU (CUDA). Install the CUDA build of torch "
            "(see desktop/README.md) and the NVIDIA driver."
        )
    return report


class DesktopEngine:
    """Answers protocol requests; models load on a background thread."""

    def __init__(self, config: DesktopConfig, load: Callable[[], Engine]) -> None:
        self.config = config
        self.store = WorldStore(config.store, config.max_worlds)
        self.runner: JobRunner | None = None
        self.state = "loading"
        self.error: str | None = None
        self.engine: dict[str, Any] = {}
        self.gpu: dict[str, Any] = {"policy": config.gpu}
        self._load = load
        self._loader = threading.Thread(target=self._start, name="engine-loader", daemon=True)
        self._loader.start()

    def _start(self) -> None:
        try:
            self.gpu = gpu_report(self.config.gpu)
            if self.config.gpu == "nvidia" and self.gpu["problems"]:
                raise RuntimeError(" ".join(self.gpu["problems"]))
            estimator, inpainter, generator, classify = self._load()
            self.runner = JobRunner(
                estimator,
                self.store,
                inpainter=inpainter,
                generator=generator,
                classify=classify,
                surround_cache=self.config.store.parent / "surround-cache" if generator is not None else None,
            )
            self.engine = {
                "depth": estimator.describe(),
                "inpainting": inpainter.describe() if inpainter is not None else None,
                "surround": generator.describe() if generator is not None else None,
            }
            self.state = "ready"
            logger.info("Engine ready: %s", self.engine)
        except Exception as exc:
            logger.exception("Engine failed to start")
            self.state, self.error = "failed", str(exc) or exc.__class__.__name__

    def wait_loaded(self, timeout: float | None = None) -> None:
        self._loader.join(timeout)

    def stop(self) -> None:
        if self.runner is not None:
            self.runner.stop()

    # -- operations --------------------------------------------------------
    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "status":
            return {
                "protocol": PROTOCOL,
                "state": self.state,
                "error": self.error,
                "gpu": self.gpu,
                "engine": self.engine,
                "store": str(self.config.store.resolve()),
                "profiles": list(PROFILES),
                "pending": self.runner.pending() if self.runner else 0,
            }
        if op == "build":
            return self._build(request)
        if op == "job":
            job_id = str(request.get("job_id", ""))
            status = self.runner.status(job_id) if self.runner and ID_PATTERN.match(job_id) else None
            if status is None:
                raise ValueError("Unknown job")
            return status
        raise ValueError(f"Unknown operation {op!r}")

    def _build(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.runner is None:
            raise RuntimeError(self.error if self.state == "failed" else "The engine is still starting")
        quality = request.get("quality") or "desktop"
        if quality not in PROFILES:
            raise ValueError(f"quality must be one of {', '.join(PROFILES)}")
        path = Path(str(request.get("path", "")))
        try:
            size = path.stat().st_size
            if size > self.config.max_upload_mb * 1024 * 1024:
                raise ValueError(f"Photo is larger than {self.config.max_upload_mb} MB")
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError("The photo could not be read") from exc
        finally:
            if request.get("delete"):
                path.unlink(missing_ok=True)  # the app's own upload copy
        if not payload:
            raise ValueError("Empty photo")
        if not _looks_like_image(payload):
            raise ValueError("That file isn't an image")
        name = _world_name(request.get("name"))
        try:
            job = self.runner.submit(payload, quality, name)
        except QueueFull as exc:
            raise RuntimeError(f"The engine is busy ({exc}); try again shortly") from exc
        return {"job_id": job.id}


def _world_name(value: Any) -> str | None:
    """The photo's file name without its extension, tidied; None if unusable."""
    if not isinstance(value, str):
        return None
    stem = Path(value.replace("\\", "/")).name.rsplit(".", 1)[0]
    stem = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", " ", stem).strip()[:80]
    return stem or None


def _protocol_stream() -> TextIO:
    """A private copy of stdout for replies; fd 1 itself now goes to stderr."""
    sys.stdout.flush()
    stream = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return stream


def serve(engine: DesktopEngine, requests: TextIO, replies: TextIO) -> None:
    """Answer each request line until ``shutdown`` or end of input."""
    for line in requests:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
        except ValueError as exc:
            replies.write(json.dumps({"id": None, "ok": False, "error": f"Bad request: {exc}"}) + "\n")
            replies.flush()
            continue
        reply: dict[str, Any] = {"id": request.get("id")}
        if request.get("op") == "shutdown":
            replies.write(json.dumps({**reply, "ok": True, "result": None}) + "\n")
            replies.flush()
            return
        try:
            reply.update(ok=True, result=engine.handle(request))
        except Exception as exc:
            if not isinstance(exc, (ValueError, RuntimeError)):
                logger.exception("Request failed")
            reply.update(ok=False, error=str(exc) or exc.__class__.__name__)
        replies.write(json.dumps(reply) + "\n")
        replies.flush()


def run(config: DesktopConfig, load: Callable[[], Engine]) -> int:
    replies = _protocol_stream()
    if config.gpu == "nvidia":
        os.environ["VIEWFINDER_GPU"] = "nvidia"  # read by render.py before its first context
    engine = DesktopEngine(config, load)
    try:
        serve(engine, sys.stdin, replies)
    finally:
        engine.stop()
    return 0
