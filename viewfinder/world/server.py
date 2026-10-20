"""HTTP server so a phone on the same network can build and explore worlds.

Stdlib only. Uploads are the raw image body (no multipart: ``cgi`` is gone in
Python 3.13). Building is asynchronous, because mobile browsers drop
long-lived requests when the screen sleeps: ``POST /api/worlds`` returns a job
id immediately and the client polls ``GET /api/jobs/<id>``. One worker thread
owns the depth model, since a second copy on the GPU would gain nothing.

Exposure: loopback by default. Binding anything else requires an access token
(generated automatically) on every ``/api/*`` call. See ARCHITECTURE.md §6.
"""

from __future__ import annotations

import gzip
import hmac
import ipaddress
import json
import logging
import queue
import re
import secrets
import shutil
import socket
import ssl
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from .depth import DepthEstimator
from .export import VIEWER_ASSETS, viewer_asset
from .ingest import ImageInputError
from .pipeline import build_world
from .profiles import PROFILES
from .render import WrongGPU

logger = logging.getLogger(__name__)

ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
# No 0/o, 1/l/i: the token is typed on a phone keyboard once (PDD-REVIEW R4).
TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
TOKEN_LENGTH = 10
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


def new_token() -> str:
    return "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(TOKEN_LENGTH))


def is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return host == "localhost"
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def lan_addresses() -> list[str]:
    """Best-effort IPv4 addresses a phone on the LAN could reach."""
    found: list[str] = []
    try:
        # Connecting a UDP socket sends nothing; it only selects the route.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in found and not address.startswith("127."):
                found.append(address)
    except OSError:
        pass
    return found


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class WorldStore:
    """Built worlds on disk, newest ``max_worlds`` kept."""

    def __init__(self, root: Path, max_worlds: int = 30) -> None:
        self.root = Path(root)
        self.max_worlds = max(1, max_worlds)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, world_id: str, name: str) -> Path | None:
        if not ID_PATTERN.match(world_id) or name not in ("world.glb", "world.glb.gz", "manifest.json"):
            return None
        candidate = self.root / world_id / name
        return candidate if candidate.is_file() else None

    def save(self, world_id: str, glb: bytes, manifest: dict[str, Any]) -> None:
        directory = self.root / world_id
        staging = self.root / f".{world_id}.tmp"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "world.glb").write_bytes(glb)
        # Precompressed once here, not per request (PDD-REVIEW R7).
        (staging / "world.glb.gz").write_bytes(gzip.compress(glb, compresslevel=6))
        (staging / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self._lock:
            staging.rename(directory)
            self._evict()

    def _evict(self) -> None:
        worlds = sorted(
            (p for p in self.root.iterdir() if p.is_dir() and ID_PATTERN.match(p.name)),
            key=lambda p: p.stat().st_mtime,
        )
        for stale in worlds[: max(0, len(worlds) - self.max_worlds)]:
            shutil.rmtree(stale, ignore_errors=True)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    quality: str
    payload: bytes | None
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    world_id: str | None = None
    error: str | None = None
    name: str | None = None  # what to call the world; the upload's bytes carry no file name
    created: float = field(default_factory=time.time)

    def as_dict(self, position: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
        }
        if self.status == "queued":
            body["position"] = position
        if self.world_id:
            body["world_id"] = self.world_id
        if self.error:
            body["error"] = self.error
        return body


class QueueFull(RuntimeError):
    pass


class JobRunner:
    """One worker thread; a bounded backlog; finished jobs remembered briefly."""

    def __init__(
        self,
        estimator: DepthEstimator,
        store: WorldStore,
        max_pending: int = 8,
        builder: Callable[..., Any] = build_world,
        inpainter: Any = None,
        generator: Any = None,
        classify: Callable[..., Any] | None = None,
        surround_cache: Path | None = None,
    ) -> None:
        self.surround_cache = surround_cache
        self.estimator = estimator
        self.inpainter = inpainter
        self.generator = generator
        self.classify = classify
        self.store = store
        self.max_pending = max_pending
        self.builder = builder
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, name="world-builder", daemon=True)
        self._thread.start()

    def submit(self, payload: bytes, quality: str, name: str | None = None) -> Job:
        with self._lock:
            pending = sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))
            if pending >= self.max_pending:
                raise QueueFull(f"{pending} photos are already waiting")
            job = Job(id=uuid.uuid4().hex, quality=quality, payload=payload, name=name)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._forget_old()
        self._queue.put(job.id)
        return job

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # Jobs ahead, the one being built included: "waiting for the
            # engine" with nothing ahead in the list looked like a hang while
            # another photo was building.
            queued = [j for j in self._order if self._jobs[j].status == "queued"]
            running = sum(1 for j in self._jobs.values() if j.status == "running")
            position = running + queued.index(job_id) if job_id in queued else 0
            return job.as_dict(position)

    def pending(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    def _forget_old(self) -> None:
        while len(self._order) > 200:
            oldest = self._order[0]
            if self._jobs[oldest].status in ("queued", "running"):
                break
            self._order.pop(0)
            self._jobs.pop(oldest, None)

    def stop(self) -> None:
        self._queue.put(None)

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self._jobs[job_id]
            job.status = "running"
            job.stage = "start"

            def progress(stage: str, fraction: float, job: Job = job) -> None:
                job.stage, job.progress = stage, fraction

            try:
                result = self.builder(
                    job.payload,
                    profile=job.quality,
                    estimator=self.estimator,
                    inpainter=self.inpainter,
                    generator=self.generator,
                    classify=self.classify,
                    progress=progress,
                    **({"surround_cache": self.surround_cache} if self.surround_cache else {}),
                    **({"name": job.name} if job.name else {}),
                )
                world_id = uuid.uuid4().hex
                self.store.save(world_id, result.glb, result.manifest)
                job.world_id, job.status, job.progress = world_id, "done", 1.0
                logger.info("Job %s -> world %s (%.1f MB)", job.id[:8], world_id[:8], len(result.glb) / 1e6)
            except (ImageInputError, WrongGPU) as exc:
                job.status, job.error = "error", str(exc)
            except Exception as exc:  # never let one photo kill the worker
                logger.exception("World build failed")
                job.status, job.error = "error", f"The world could not be built ({exc.__class__.__name__})."
            finally:
                job.payload = None  # free the upload
                release = getattr(self.generator, "release_memory", None)
                if release is not None:
                    release()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class WorldHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        runner: JobRunner,
        store: WorldStore,
        *,
        token: str | None,
        max_upload_bytes: int,
        default_quality: str,
    ) -> None:
        super().__init__(address, WorldRequestHandler)
        self.runner = runner
        self.store = store
        self.token = token
        self.max_upload_bytes = max_upload_bytes
        self.default_quality = default_quality


class WorldRequestHandler(BaseHTTPRequestHandler):
    server: WorldHTTPServer
    server_version = "ViewFinderWorld/1"

    # -- plumbing ------------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # Never log the query string: it can carry the access token.
        logger.debug("%s %s -> %s", self.command, urlsplit(self.path).path, args[1] if len(args) > 1 else "")

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(
            status,
            json.dumps(payload).encode("utf-8"),
            "application/json",
            {"Cache-Control": "no-store"},
        )

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _authorized(self, query: dict[str, list[str]]) -> bool:
        expected = self.server.token
        if not expected:
            return True
        supplied = self.headers.get("X-World-Token") or (query.get("t") or [""])[0]
        return hmac.compare_digest(supplied.encode(), expected.encode())

    # -- routes ----------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        path, query = parts.path, parse_qs(parts.query)
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/") :])
        if path == "/favicon.ico":
            return self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        if not path.startswith("/api/"):
            return self._error(HTTPStatus.NOT_FOUND, "Not found")
        if not self._authorized(query):
            return self._error(HTTPStatus.UNAUTHORIZED, "Access token missing or wrong")
        if path == "/api/health":
            return self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "depth": self.server.runner.estimator.describe(),
                    "inpainting": self.server.runner.inpainter.describe() if self.server.runner.inpainter else None,
                    "surround": self.server.runner.generator.describe() if self.server.runner.generator else None,
                    "pending": self.server.runner.pending(),
                    "profiles": list(PROFILES),
                },
            )
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", path)
        if match:
            status = self.server.runner.status(match.group(1))
            if status is None:
                return self._error(HTTPStatus.NOT_FOUND, "Unknown job")
            return self._json(HTTPStatus.OK, status)
        match = re.fullmatch(r"/api/worlds/([0-9a-f]{32})/(world\.glb|manifest\.json)", path)
        if match:
            return self._world_file(match.group(1), match.group(2))
        return self._error(HTTPStatus.NOT_FOUND, "Not found")

    def _reject_post(self, status: int, message: str) -> None:
        """Reply with an error to an upload we will not process.

        The body is drained first (when plausibly sized): closing a socket
        with unread data makes Windows, and many clients, report a
        connection reset instead of the response. A phone with a stale token
        would then see a network failure rather than "token missing".
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if 0 < length <= 4 * self.server.max_upload_bytes:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 16))
                if not chunk:
                    break
                remaining -= len(chunk)
        self.close_connection = True
        self._error(status, message)

    def do_POST(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        if parts.path != "/api/worlds":
            return self._reject_post(HTTPStatus.NOT_FOUND, "Not found")
        if not self._authorized(query):
            return self._reject_post(HTTPStatus.UNAUTHORIZED, "Access token missing or wrong")
        quality = (query.get("quality") or [self.server.default_quality])[0]
        if quality not in PROFILES:
            return self._reject_post(HTTPStatus.BAD_REQUEST, f"quality must be one of {', '.join(PROFILES)}")
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return self._reject_post(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        try:
            length = int(length_header)
        except ValueError:
            return self._reject_post(HTTPStatus.BAD_REQUEST, "Bad Content-Length")
        if length <= 0:
            return self._reject_post(HTTPStatus.BAD_REQUEST, "Empty upload")
        if length > self.server.max_upload_bytes:
            # Bodies up to 4x the limit are drained so the client receives the
            # 413; absurd sizes are not read at all.
            return self._reject_post(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"Photo is larger than {self.server.max_upload_bytes // (1024 * 1024)} MB",
            )
        payload = self.rfile.read(length)
        if len(payload) != length:
            return self._error(HTTPStatus.BAD_REQUEST, "Upload was cut off")
        if not _looks_like_image(payload):
            return self._error(HTTPStatus.BAD_REQUEST, "That file isn't an image")
        try:
            job = self.server.runner.submit(payload, quality)
        except QueueFull as exc:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, f"The engine is busy ({exc}); try again shortly")
        self._json(
            HTTPStatus.ACCEPTED,
            {"job_id": job.id, "status_url": f"/api/jobs/{job.id}"},
        )

    # -- helpers -------------------------------------------------------------
    def _static(self, name: str) -> None:
        if name not in VIEWER_ASSETS:
            return self._error(HTTPStatus.NOT_FOUND, "Not found")
        body = viewer_asset(name)
        content_type = CONTENT_TYPES.get(Path(name).suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, body, content_type, {"Cache-Control": "no-cache"})

    def _world_file(self, world_id: str, name: str) -> None:
        if name == "manifest.json":
            path = self.server.store.path(world_id, name)
            if path is None:
                return self._error(HTTPStatus.NOT_FOUND, "Unknown world")
            return self._send(HTTPStatus.OK, path.read_bytes(), "application/json")
        accepts_gzip = "gzip" in (self.headers.get("Accept-Encoding") or "")
        gz = self.server.store.path(world_id, "world.glb.gz") if accepts_gzip else None
        path = gz or self.server.store.path(world_id, "world.glb")
        if path is None:
            return self._error(HTTPStatus.NOT_FOUND, "Unknown world")
        headers = {"Cache-Control": "private, max-age=31536000, immutable", "Vary": "Accept-Encoding"}
        if gz:
            headers["Content-Encoding"] = "gzip"
        self._send(HTTPStatus.OK, path.read_bytes(), "model/gltf-binary", headers)


# Leading bytes of the formats a phone or desktop will realistically send.
_SIGNATURES = (
    b"\xff\xd8\xff",  # JPEG
    b"\x89PNG\r\n\x1a\n",
    b"GIF8",
    b"BM",
    b"II*\x00",  # TIFF
    b"MM\x00*",
)


def _looks_like_image(payload: bytes) -> bool:
    """Cheap pre-check before queueing; the decoder remains the real gate."""
    head = payload[:32]
    if head.startswith(_SIGNATURES):
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    # ISO-BMFF images: HEIC/HEIF/AVIF carry "ftyp" at offset 4.
    return head[4:8] == b"ftyp"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


@dataclass
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    store: Path = Path("local-data/worlds")
    max_worlds: int = 30
    max_upload_mb: int = 25
    max_pending: int = 8
    default_quality: str = "mobile"
    token: str | None = None  # None -> auto: required off-loopback
    require_token: bool | None = None
    tls_cert: Path | None = None
    tls_key: Path | None = None


def create_server(
    config: ServeConfig,
    estimator: DepthEstimator,
    inpainter: Any = None,
    generator: Any = None,
    classify: Callable[..., Any] | None = None,
) -> WorldHTTPServer:
    needs_token = config.require_token if config.require_token is not None else not is_loopback(config.host)
    token = (config.token or new_token()) if needs_token else None
    store = WorldStore(config.store, config.max_worlds)
    runner = JobRunner(
        estimator,
        store,
        max_pending=config.max_pending,
        inpainter=inpainter,
        generator=generator,
        classify=classify,
        # Next to the world store: re-uploading a photo skips generation.
        surround_cache=Path(config.store).parent / "surround-cache" if generator is not None else None,
    )
    server = WorldHTTPServer(
        (config.host, config.port),
        runner,
        store,
        token=token,
        max_upload_bytes=config.max_upload_mb * 1024 * 1024,
        default_quality=config.default_quality,
    )
    if config.tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(config.tls_cert), str(config.tls_key) if config.tls_key else None)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


def server_urls(server: WorldHTTPServer, config: ServeConfig) -> list[str]:
    scheme = "https" if config.tls_cert else "http"
    port = server.server_address[1]
    hosts = lan_addresses() if config.host in ("0.0.0.0", "::", "") else [config.host]
    suffix = f"/?t={server.token}" if server.token else "/"
    return [f"{scheme}://{host}:{port}{suffix}" for host in hosts or ["localhost"]]
