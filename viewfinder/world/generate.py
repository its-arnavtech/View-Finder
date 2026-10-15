"""Generative surround: complete the 360° panorama around a photo.

The photo is placed on an equirectangular canvas at its true field of view
(panorama.py). Everything else is generated through a ring of perspective
views, each inpainted by an image model with what is already known (the
photo, then earlier views) as its context, then pasted back. Going round the
horizon first and closing the loop behind the camera gives one continuous
world; the poles (sky or ceiling, ground or floor) are filled last.

The model is a pluggable *backend*: anything that can inpaint a masked RGB
view from a prompt. The local backend is Stable Diffusion XL inpainting on
the GPU; a hosted model can implement the same three methods.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import cv2
import numpy as np

from .layers import propagate_background
from .panorama import View, blend_view, render_view
from .scene import Scene

logger = logging.getLogger(__name__)

SDXL_INPAINT = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SDXL_VAE = "madebyollin/sdxl-vae-fp16-fix"
SDXL_FILES = (
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
)
VAE_FILES = ("config.json", "diffusion_pytorch_model.safetensors")


class GeneratorUnavailable(RuntimeError):
    """The generative model, its runtime or a GPU is missing."""


class ViewBackend(Protocol):
    name: str

    def inpaint(
        self, image: np.ndarray, mask: np.ndarray, prompt: str, negative: str, seed: int, strength: float = 0.99
    ) -> np.ndarray:
        """Fill ``mask`` (True = generate) in the square RGB ``image``. A low
        ``strength`` keeps the image's structure and only adds detail."""
        ...

    def describe(self) -> dict[str, str]: ...


class SDXLBackend:
    """Stable Diffusion XL inpainting (CreativeML Open RAIL++-M) on the local GPU."""

    name = "sdxl-inpaint"

    def __init__(self, device: str = "auto", steps: int = 24, guidance: float = 7.0) -> None:
        try:
            import torch
            from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline
        except ImportError as exc:
            raise GeneratorUnavailable(
                "diffusers is not installed; install viewfinder/requirements-world.txt"
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # Minutes per view on a CPU: not a usable default.
            raise GeneratorUnavailable("the surround generator needs an NVIDIA GPU (CUDA)")
        if not models_cached():
            raise GeneratorUnavailable(
                "The surround model is not installed. Run `view-finder world setup` once with network "
                "access to download it (~7 GB)."
            )
        logger.info("Loading the surround generator (SDXL inpainting)…")
        self._torch = torch
        vae = AutoencoderKL.from_pretrained(SDXL_VAE, torch_dtype=torch.float16, local_files_only=True)
        full = StableDiffusionXLInpaintPipeline.from_pretrained(
            SDXL_INPAINT, vae=vae, torch_dtype=torch.float16, variant="fp16", local_files_only=True
        )
        # The UNet (~5 GB in fp16) and VAE live on the GPU for good. The text
        # encoders (~1.6 GB) stay in RAM and visit the GPU only to encode a
        # new prompt; a world uses three. diffusers' model CPU offload instead
        # moved the UNet in and out of RAM on every view, which needed ~13 GB
        # of RAM and paged to a crawl on a 24 GB laptop.
        self._encoders = (full.text_encoder, full.text_encoder_2)
        self._encode = full.encode_prompt
        full.unet.to(device)
        full.vae.to(device)
        # The VAE decodes in overlapping 512-pixel tiles. Decoding a whole
        # 1024 view at once was the peak that, on top of the resident UNet,
        # spilled the 8 GB card into system memory: a station that takes four
        # minutes ran for twenty.
        full.vae.enable_tiling()
        full.vae.tile_sample_min_size = 512
        full.vae.tile_latent_min_size = 64
        self._pipe = StableDiffusionXLInpaintPipeline(
            **{**full.components, "text_encoder": None, "text_encoder_2": None}
        )
        # DPM++ 2M with Karras sigmas: crisper detail than the default Euler
        # sampler at the same step count.
        from diffusers import DPMSolverMultistepScheduler

        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )
        self._pipe.set_progress_bar_config(disable=True)
        self._embeddings: dict[tuple[str, str], tuple] = {}
        self.device = device
        self.steps = steps
        self.guidance = guidance
        self._lock = threading.Lock()

    def _embed(self, prompt: str, negative: str) -> tuple:
        key = (prompt, negative)
        if key not in self._embeddings:
            torch = self._torch
            for encoder in self._encoders:
                encoder.to(self.device)
            with torch.inference_mode():
                self._embeddings[key] = self._encode(
                    prompt=prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative,
                )
            for encoder in self._encoders:
                encoder.to("cpu")
            torch.cuda.empty_cache()
            if len(self._embeddings) > 16:
                self._embeddings.pop(next(iter(self._embeddings)))
        return self._embeddings[key]

    def inpaint(
        self, image: np.ndarray, mask: np.ndarray, prompt: str, negative: str, seed: int, strength: float = 0.99
    ) -> np.ndarray:
        from PIL import Image

        size = image.shape[0]
        torch = self._torch
        with self._lock:
            embeds, negative_embeds, pooled, negative_pooled = self._embed(prompt, negative)
            # The UNet nearly fills an 8 GB card. Windows usually spills past
            # that into system memory, but when RAM is short too an
            # allocation fails outright; freeing the allocator's cache and
            # trying once more gets through.
            for attempt in (1, 2):
                try:
                    result = self._pipe(
                        prompt_embeds=embeds,
                        negative_prompt_embeds=negative_embeds,
                        pooled_prompt_embeds=pooled,
                        negative_pooled_prompt_embeds=negative_pooled,
                        image=Image.fromarray(image),
                        mask_image=Image.fromarray(mask.astype(np.uint8) * 255),
                        height=size,
                        width=size,
                        num_inference_steps=self.steps,
                        guidance_scale=self.guidance,
                        strength=strength,
                        generator=torch.Generator("cpu").manual_seed(seed),
                    ).images[0]
                    break
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if attempt == 2 or "out of memory" not in str(exc):
                        raise
                    logger.warning("The GPU ran out of memory; freeing its cache and retrying")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            torch.cuda.empty_cache()  # no fragmentation building up view after view
        return np.asarray(result.convert("RGB").resize((size, size)))

    def release_memory(self) -> None:
        """Return cached GPU memory between worlds."""
        import gc

        gc.collect()
        self._torch.cuda.empty_cache()

    def describe(self) -> dict[str, str]:
        return {"generator": self.name, "model": SDXL_INPAINT, "device": self.device}


def models_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return False
    needed = [(SDXL_INPAINT, "unet/diffusion_pytorch_model.fp16.safetensors"),
              (SDXL_INPAINT, "text_encoder_2/model.fp16.safetensors"),
              (SDXL_VAE, "diffusion_pytorch_model.safetensors")]
    return all(isinstance(try_to_load_from_cache(repo, name), str) for repo, name in needed)


def download_models() -> list[str]:
    from huggingface_hub import snapshot_download

    # One file at a time: parallel downloads into the cache race on Windows
    # when it cannot create symlinks.
    return [
        snapshot_download(SDXL_INPAINT, allow_patterns=list(SDXL_FILES), max_workers=1),
        snapshot_download(SDXL_VAE, allow_patterns=list(VAE_FILES), max_workers=1),
    ]


_cache: dict[str, SDXLBackend] = {}
_cache_lock = threading.Lock()


def load_backend(device: str = "auto") -> SDXLBackend:
    """Process-cached local backend; raises GeneratorUnavailable."""
    with _cache_lock:
        if device not in _cache:
            _cache[device] = SDXLBackend(device)
        return _cache[device]


# -- panorama completion ------------------------------------------------------

#: Horizon ring first (nearest the photo outward, closing behind the camera),
#: then a ring tilted up, then straight down. 100° views at 60° steps
#: overlap by 40°.
HORIZON = (0.0, 60.0, -60.0, 120.0, -120.0, 180.0)
#: Four views tilted up by this much reach the zenith together. A single
#: view straight up saw tree and mountain tops all round its edge and
#: continued them across the sky as an overhanging rock ceiling; a tilted
#: view looks like an ordinary photo with the treeline at the bottom and
#: open sky (or a ceiling, indoors) above.
UPWARD_PITCH = 50.0
UPWARD = (0.0, 90.0, 180.0, -90.0)
VIEW_FOV = 100.0


@dataclass
class Completion:
    panorama: np.ndarray  # (H, W, 3) uint8, photo + generated surround
    generated: np.ndarray  # (H, W) bool, pixels the model invented
    views: list[View] = field(default_factory=list)  # the views actually generated
    seconds: float = 0.0


def view_plan(size: int) -> list[View]:
    views = [View(yaw, 0.0, VIEW_FOV, size) for yaw in HORIZON]
    views += [View(yaw, UPWARD_PITCH, VIEW_FOV, size) for yaw in UPWARD]
    views.append(View(0.0, -90.0, VIEW_FOV, size))
    return views


def complete_panorama(
    panorama: np.ndarray,
    known: np.ndarray,
    scene: Scene,
    backend: ViewBackend,
    view_size: int = 1024,
    progress: Callable[[int, int], None] | None = None,
    seed: int = 0,
    protected: np.ndarray | None = None,
) -> Completion:
    """Fill every unknown pixel of ``panorama`` (modified in place).

    ``protected`` pixels are never repainted (default: everything initially
    known). Known pixels that are not protected, such as a thin ring just
    inside the photo's border, are blended with the model's repaint, so
    new content fades into the photo instead of meeting it at a seam.
    """
    started = time.perf_counter()
    known = known.copy()
    original = known.copy()
    protected = original if protected is None else protected
    plan = view_plan(view_size)
    done: list[View] = []
    rng = np.random.default_rng(seed)
    for index, view in enumerate(plan):
        image = render_view(panorama, view)
        # Known only where every interpolated neighbour is known, minus a
        # pixel: at the boundary the view's bilinear sampling mixes in the
        # unfilled (black) canvas, and the model continued that aliased
        # zigzag as comb-like stripes.
        known_view = render_view(known.astype(np.float32), view) > 0.999
        known_view = cv2.erode(known_view.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        hole = ~known_view
        if hole.mean() > 0.003:
            direction = "up" if view.pitch_deg > 30 else ("down" if view.pitch_deg < -30 else "around")
            # The model also repaints a band (~5°) of known pixels around the
            # hole, and that band is blended back with a weight falling from
            # 1 at the hole to 0 at its outer edge, so new content fades into
            # the old instead of meeting it at a seam.
            band = max(4, view_size // 20)
            distance = cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 5)
            grown = distance <= band
            # Seed the hole with a soft continuation of what is known, plus
            # fine noise. A black hole kept the result black (the model starts
            # from a trace of its input), pure noise read the photo as a
            # framed picture hanging in a room, and a smooth seed alone made
            # each view as soft as its seed: haze that grew away from the photo.
            image = propagate_background(image, hole, known_view)
            noise = rng.normal(0.0, 12.0, image.shape).astype(np.float32)
            image = np.where(hole[..., None], np.clip(image + noise, 0, 255), image).astype(np.uint8)
            generated = backend.inpaint(image, grown, scene.prompt(direction), scene.negative(direction), seed + index)
            generated = match_colours(generated, image, grown & known_view)
            # Over the photo itself (only its edge ring is unprotected) the
            # fade is twice as short, so it is complete within the ring.
            photo_view = render_view(original.astype(np.uint8), view, cv2.INTER_NEAREST).astype(bool)
            ramp = np.clip(1.0 - distance / np.where(photo_view, band / 2.0, band), 0.0, 1.0)
            soft = (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)
            merged = (image * (1.0 - soft[..., None]) + generated * soft[..., None]).astype(np.uint8)
            blend_view(panorama, known, protected, merged, soft, view)
            done.append(view)
            logger.debug("view yaw %+.0f pitch %+.0f: %.0f%% generated", view.yaw_deg, view.pitch_deg, 100 * hole.mean())
        if progress:
            progress(index + 1, len(plan))
    if not known.all():
        # View borders the ring never revisited: a few thin slivers at most.
        panorama[:] = propagate_background(panorama, ~known, known)
    return Completion(panorama, ~original, done, time.perf_counter() - started)


def match_colours(generated: np.ndarray, reference: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Per-channel gain and offset that make ``generated`` match ``reference``
    over ``region`` (known pixels the model also repainted). Image models
    drift in brightness and tint from view to view; uncorrected, each view
    shows as a panel."""
    if region.sum() < 100:
        return generated
    out = generated.astype(np.float32)
    for channel in range(3):
        g = out[..., channel][region]
        r = reference[..., channel][region].astype(np.float32)
        gain = float(np.clip(r.std() / max(g.std(), 1.0), 0.75, 1.33))
        offset = float(r.mean() - gain * g.mean())
        out[..., channel] = out[..., channel] * gain + offset
    return np.clip(out, 0, 255).astype(np.uint8)
