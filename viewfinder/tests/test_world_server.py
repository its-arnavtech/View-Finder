import gzip
import http.client
import json
import threading
import time

import pytest

from viewfinder.world.gltf import read_glb
from viewfinder.world.server import (
    TOKEN_ALPHABET,
    ServeConfig,
    WorldStore,
    create_server,
    is_loopback,
    new_token,
)

from world_fixtures import SceneEstimator, scene_jpeg

TOKEN = "abcdefghjk"


@pytest.fixture()
def server(tmp_path):
    config = ServeConfig(
        host="127.0.0.1",
        port=0,
        store=tmp_path / "worlds",
        max_upload_mb=1,
        require_token=True,
        token=TOKEN,
    )
    instance = create_server(config, SceneEstimator())
    thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.runner.stop()
    instance.server_close()


def request(server, method, path, body=None, headers=None, token=TOKEN):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
    headers = dict(headers or {})
    if token:
        headers["X-World-Token"] = token
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response, data


def wait_for_job(server, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, data = request(server, "GET", f"/api/jobs/{job_id}")
        status = json.loads(data)
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_upload_poll_and_download_a_world(server):
    response, data = request(
        server, "POST", "/api/worlds?quality=mobile", scene_jpeg(), {"Content-Type": "image/jpeg"}
    )
    assert response.status == 202
    status = wait_for_job(server, json.loads(data)["job_id"])
    assert status["status"] == "done" and status["progress"] == 1.0
    world_id = status["world_id"]

    response, glb = request(server, "GET", f"/api/worlds/{world_id}/world.glb")
    assert response.status == 200 and response.getheader("Content-Type") == "model/gltf-binary"
    assert "immutable" in response.getheader("Cache-Control")
    assert read_glb(glb).extras["viewfinder_world"]["profile"] == "mobile"

    response, zipped = request(
        server, "GET", f"/api/worlds/{world_id}/world.glb", headers={"Accept-Encoding": "gzip"}
    )
    assert response.getheader("Content-Encoding") == "gzip"
    assert gzip.decompress(zipped) == glb

    response, manifest = request(server, "GET", f"/api/worlds/{world_id}/manifest.json")
    assert json.loads(manifest)["layers"]["foreground"]["triangles"] > 0


def test_token_is_required_and_accepted_as_query(server):
    response, _ = request(server, "GET", "/api/health", token=None)
    assert response.status == 401
    response, _ = request(server, "GET", "/api/health", token="wrongtoken")
    assert response.status == 401
    response, data = request(server, "GET", f"/api/health?t={TOKEN}", token=None)
    assert response.status == 200 and json.loads(data)["depth"]["estimator"] == "synthetic"


def test_viewer_assets_are_public_and_whitelisted(server):
    response, html = request(server, "GET", "/", token=None)
    assert response.status == 200 and b"static/app.js" in html
    response, _ = request(server, "GET", "/static/renderer.js", token=None)
    assert response.status == 200 and "javascript" in response.getheader("Content-Type")
    for path in ("/static/../server.py", "/static/cli.py", "/etc/passwd"):
        assert request(server, "GET", path, token=None)[0].status == 404


def test_upload_validation(server):
    too_big = b"\xff\xd8\xff" + b"\0" * (1024 * 1024 + 10)
    assert request(server, "POST", "/api/worlds", too_big)[0].status == 413
    assert request(server, "POST", "/api/worlds", b"hello world")[0].status == 400
    assert request(server, "POST", "/api/worlds?quality=ultra", scene_jpeg())[0].status == 400
    assert request(server, "POST", "/api/nope", scene_jpeg())[0].status == 404


def test_corrupt_image_becomes_a_job_error(server):
    _, data = request(server, "POST", "/api/worlds", b"\xff\xd8\xff\xe0 but truncated")
    status = wait_for_job(server, json.loads(data)["job_id"])
    assert status["status"] == "error" and "readable image" in status["error"]


def test_ids_are_validated(server):
    assert request(server, "GET", "/api/jobs/not-a-hex-id")[0].status == 404
    assert request(server, "GET", "/api/jobs/" + "0" * 32)[0].status == 404
    assert request(server, "GET", "/api/worlds/" + "0" * 32 + "/world.glb")[0].status == 404
    assert request(server, "GET", "/api/worlds/" + "0" * 32 + "/secret.txt")[0].status == 404


def test_store_keeps_only_the_newest_worlds(tmp_path):
    store = WorldStore(tmp_path, max_worlds=2)
    ids = [f"{i:032x}" for i in range(3)]
    for world_id in ids:
        store.save(world_id, b"glb", {"id": world_id})
        time.sleep(0.02)
    assert store.path(ids[0], "world.glb") is None
    assert store.path(ids[2], "world.glb.gz") is not None
    assert store.path("../" + ids[2], "world.glb") is None


def test_queue_is_bounded(tmp_path):
    import threading as _threading

    from viewfinder.world.server import JobRunner, QueueFull

    gate = _threading.Event()

    def slow_builder(*args, **kwargs):
        gate.wait(5)
        raise RuntimeError("stop")

    runner = JobRunner(SceneEstimator(), WorldStore(tmp_path), max_pending=2, builder=slow_builder)
    runner.submit(b"x", "mobile")
    runner.submit(b"y", "mobile")
    with pytest.raises(QueueFull):
        runner.submit(b"z", "mobile")
    gate.set()
    runner.stop()


def test_token_shape_and_loopback_detection():
    token = new_token()
    assert len(token) == 10 and set(token) <= set(TOKEN_ALPHABET)
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5") and not is_loopback("")


def test_lan_binding_requires_a_token_by_default(tmp_path):
    instance = create_server(ServeConfig(host="0.0.0.0", port=0, store=tmp_path), SceneEstimator())
    try:
        assert instance.token and len(instance.token) == 10
    finally:
        instance.runner.stop()
        instance.server_close()


def test_rejected_uploads_still_deliver_their_error(server):
    """Every refusal must reach the client as a status, not a reset. A wrong
    token on a phone must read "token missing", not a network failure."""
    body = scene_jpeg()
    for _ in range(5):
        assert request(server, "POST", "/api/worlds", body, token="wrongtoken")[0].status == 401
        assert request(server, "POST", "/api/nope", body)[0].status == 404
        assert request(server, "POST", "/api/worlds?quality=ultra", body)[0].status == 400
