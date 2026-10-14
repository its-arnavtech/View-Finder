"""Write a built world to disk: GLB, manifest JSON, optional standalone HTML."""

from __future__ import annotations

import base64
import json
import logging
import re
from importlib import resources
from pathlib import Path

from .pipeline import WorldResult

logger = logging.getLogger(__name__)

VIEWER_ASSETS = ("index.html", "viewer.css", "glb.js", "renderer.js", "controls.js", "app.js")
# Past this, a phone browser struggles to even open the file (PDD-REVIEW R10).
STANDALONE_WARN_BYTES = 25 * 1024 * 1024


def viewer_asset(name: str) -> bytes:
    if name not in VIEWER_ASSETS:
        raise FileNotFoundError(name)
    return resources.files(__package__).joinpath("viewer", name).read_bytes()


def standalone_html(result: WorldResult) -> str:
    """The viewer with every asset and the GLB inlined: one file, no server."""
    html = viewer_asset("index.html").decode("utf-8")
    css = viewer_asset("viewer.css").decode("utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="static/viewer.css">', f"<style>\n{css}\n</style>"
    )
    payload = base64.b64encode(result.glb).decode("ascii")
    title = re.sub(r"[<>&\"']", "", str(result.manifest.get("name", "world")))
    html = html.replace("<title>View Finder World</title>", f"<title>{title} · View Finder World</title>")

    def inline(match: re.Match[str]) -> str:
        source = viewer_asset(match.group(1)).decode("utf-8")
        # A literal "</script" inside JS would end the inline block early.
        return "<script>\n" + source.replace("</script", "<\\/script") + "\n</script>"

    first_script = '<script src="static/glb.js"></script>'
    html = html.replace(
        first_script,
        f'<script>window.__VIEWFINDER_WORLD__="{payload}";</script>\n{first_script}',
        1,
    )
    return re.sub(r'<script src="static/([\w.]+)"></script>', inline, html)


def write_world(
    result: WorldResult,
    output_dir: Path,
    stem: str,
    *,
    html: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "glb": output_dir / f"{stem}_world.glb",
        "manifest": output_dir / f"{stem}_world.json",
    }
    paths["glb"].write_bytes(result.glb)
    paths["manifest"].write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")
    if html:
        paths["html"] = output_dir / f"{stem}_world.html"
        document = standalone_html(result)
        paths["html"].write_text(document, encoding="utf-8")
        if len(document) > STANDALONE_WARN_BYTES:
            logger.warning(
                "Standalone HTML is %.0f MB; phones may not open it. Use --quality mobile to share.",
                len(document) / 1e6,
            )
    return paths
