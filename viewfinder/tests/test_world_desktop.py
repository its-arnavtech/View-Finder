import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from viewfinder.world import desktop
from viewfinder.world.desktop import DesktopConfig, DesktopEngine, serve
from viewfinder.world.gltf import read_glb

from world_fixtures import SceneEstimator, scene_jpeg

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def no_gpu_probe(monkeypatch):
    monkeypatch.setattr(desktop, "gpu_report", lambda policy: {"policy": policy, "cuda": None, "renderer": None, "problems": []})


def engine_for(tmp_path, policy="any"):
    engine = DesktopEngine(
        DesktopConfig(store=tmp_path / "worlds", max_upload_mb=1, gpu=policy),
        lambda: (SceneEstimator(), None, None, None),
    )
    engine.wait_loaded(10)
    return engine


def talk(engine, *requests):
    lines = "".join(json.dumps(r) + "\n" for r in requests)
    replies = io.StringIO()
    serve(engine, io.StringIO(lines), replies)
    return [json.loads(line) for line in replies.getvalue().splitlines()]


def wait_done(engine, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = engine.handle({"op": "job", "job_id": job_id})
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_status_then_build_a_world(tmp_path, no_gpu_probe):
    engine = engine_for(tmp_path)
    photo = tmp_path / "upload.jpg"
    photo.write_bytes(scene_jpeg())
    status, built = talk(
        engine,
        {"id": 1, "op": "status"},
        {"id": 2, "op": "build", "path": str(photo), "quality": "mobile", "delete": True},
    )
    assert status["id"] == 1 and status["ok"] and status["result"]["state"] == "ready"
    assert status["result"]["engine"]["depth"]["estimator"]
    assert built["ok"], built
    assert not photo.exists()  # the app's upload copy is removed once read
    job = wait_done(engine, built["result"]["job_id"])
    assert job["status"] == "done", job
    world = tmp_path / "worlds" / job["world_id"]
    assert read_glb((world / "world.glb").read_bytes()).document["meshes"]
    assert json.loads((world / "manifest.json").read_text())["profile"] == "mobile"
    engine.stop()


def test_bad_requests_get_errors_not_a_crash(tmp_path, no_gpu_probe):
    engine = engine_for(tmp_path)
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    replies = talk(
        engine,
        {"id": 1, "op": "build", "path": str(text)},
        {"id": 2, "op": "build", "path": str(tmp_path / "missing.jpg")},
        {"id": 3, "op": "build", "path": str(text), "quality": "huge"},
        {"id": 4, "op": "job", "job_id": "0" * 32},
        {"id": 5, "op": "explode"},
    )
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5]
    assert not any(r["ok"] for r in replies)
    assert "isn't an image" in replies[0]["error"]
    assert "could not be read" in replies[1]["error"]
    assert "quality" in replies[2]["error"]
    engine.stop()


def test_oversized_photos_are_refused(tmp_path, no_gpu_probe):
    engine = engine_for(tmp_path)
    big = tmp_path / "big.jpg"
    big.write_bytes(scene_jpeg() + b"\0" * (1024 * 1024))
    (reply,) = talk(engine, {"id": 1, "op": "build", "path": str(big)})
    assert not reply["ok"] and "larger than 1 MB" in reply["error"]
    engine.stop()


def test_garbage_lines_and_shutdown(tmp_path, no_gpu_probe):
    engine = engine_for(tmp_path)
    replies = io.StringIO()
    serve(engine, io.StringIO('not json\n[1]\n{"id": 7, "op": "shutdown"}\n{"id": 8, "op": "status"}\n'), replies)
    lines = [json.loads(line) for line in replies.getvalue().splitlines()]
    assert [line["ok"] for line in lines] == [False, False, True]
    assert lines[2]["id"] == 7  # nothing is answered after shutdown
    engine.stop()


def test_nvidia_policy_refuses_to_start_without_the_nvidia_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        desktop, "gpu_report",
        lambda policy: {"policy": policy, "cuda": None, "renderer": "Intel(R) UHD Graphics",
                        "problems": ["PyTorch cannot see an NVIDIA GPU (CUDA)."]},
    )
    loaded = []
    engine = DesktopEngine(DesktopConfig(store=tmp_path / "worlds", gpu="nvidia"), lambda: loaded.append(1))
    engine.wait_loaded(10)
    assert engine.state == "failed" and "NVIDIA" in engine.error
    assert not loaded  # no model was loaded onto another device
    photo = tmp_path / "upload.jpg"
    photo.write_bytes(scene_jpeg())
    (reply,) = talk(engine, {"id": 1, "op": "build", "path": str(photo)})
    assert not reply["ok"] and "NVIDIA" in reply["error"]


def test_gpu_report_requires_cuda_for_the_nvidia_policy(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = desktop.gpu_report("nvidia")
    assert report["cuda"] is None and any("CUDA" in p for p in report["problems"])
    assert not [p for p in desktop.gpu_report("any")["problems"] if "CUDA" in p]


def test_the_process_speaks_only_protocol_on_stdout(tmp_path):
    """The real child process the app starts: stray prints go to stderr, and
    closing stdin (the app quit) ends it."""
    photo = tmp_path / "upload.jpg"
    photo.write_bytes(scene_jpeg())
    env = {**os.environ, "PYTHONPATH": str(REPO), "VIEWFINDER_WORLD_LOCATE": "0"}
    process = subprocess.Popen(
        [sys.executable, "-m", "viewfinder.world.cli", "desktop", "--store", str(tmp_path / "worlds"),
         "--gpu", "any", "--depth-model", "heuristic", "--inpaint", "none", "--generate", "off"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=tmp_path,
    )

    def ask(request):
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        return json.loads(process.stdout.readline())

    try:
        deadline = time.time() + 120
        while (status := ask({"id": 1, "op": "status"})["result"])["state"] == "loading":
            assert time.time() < deadline
            time.sleep(0.1)
        assert status["state"] == "ready", status
        assert status["gpu"]["policy"] == "any"
        job_id = ask({"id": 2, "op": "build", "path": str(photo), "quality": "mobile"})["result"]["job_id"]
        while (job := ask({"id": 3, "op": "job", "job_id": job_id})["result"])["status"] not in ("done", "error"):
            assert time.time() < deadline
            time.sleep(0.1)
        assert job["status"] == "done", job
        assert (tmp_path / "worlds" / job["world_id"] / "world.glb").is_file()
        process.stdin.close()
        assert process.wait(timeout=30) == 0
        assert process.stdout.read() == ""  # every stdout line was a reply
    finally:
        if process.poll() is None:
            process.kill()
