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
#: Depth guidance (ControlNet, CreativeML Open RAIL++-M, 0.55 GB in fp16): a
#: view is generated to follow a depth map, so a walking spot can be painted
#: afresh along the world's own geometry instead of copying a distorted
#: render of it. Optional: without it the generator works unguided.
SDXL_DEPTH_CONTROL = "diffusers/controlnet-depth-sdxl-1.0-mid"
CONTROL_FILES = ("config.json", "diffusion_pytorch_model.fp16.safetensors")
#: How firmly a view follows its depth map, and for what share of the
#: denoising steps: the layout, not every bump of a depth map rendered from
#: another viewpoint.
CONTROL_SCALE = 0.6
CONTROL_END = 0.8


class GeneratorUnavailable(RuntimeError):
    """The generative model, its runtime or a GPU is missing."""


class ViewBackend(Protocol):
    name: str

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative: str,
        seed: int,
        strength: float = 0.99,
        strength_map: np.ndarray | None = None,
        control_depth: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fill ``mask`` (True = generate) in the square RGB ``image``. A low
        ``strength`` keeps the image's structure and only adds detail;
        ``strength_map`` (0..1 per pixel) sets it pixel by pixel instead. A
        backend that cannot vary it may use the map's maximum.
        ``control_depth`` (uint8, near = bright) is the view's geometry to
        follow, used if the backend has ``depth_control``."""
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
        # Depth guidance shares every module but its own ControlNet.
        self._control_pipe = None
        self.depth_control = False
        if control_cached():
            from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

            controlnet = ControlNetModel.from_pretrained(
                SDXL_DEPTH_CONTROL, torch_dtype=torch.float16, variant="fp16", local_files_only=True
            ).to(device)
            self._control_pipe = StableDiffusionXLControlNetInpaintPipeline(
                **{**self._pipe.components, "controlnet": controlnet}
            )
            self._control_pipe.set_progress_bar_config(disable=True)
            self.depth_control = True
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
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative: str,
        seed: int,
        strength: float = 0.99,
        strength_map: np.ndarray | None = None,
        control_depth: np.ndarray | None = None,
    ) -> np.ndarray:
        from PIL import Image

        size = image.shape[0]
        pipe, control = self._pipe, {}
        if control_depth is not None and self._control_pipe is not None:
            depth = control_depth if control_depth.ndim == 3 else np.repeat(control_depth[..., None], 3, axis=2)
            pipe = self._control_pipe
            control = {
                "control_image": Image.fromarray(np.ascontiguousarray(depth.astype(np.uint8))),
                "controlnet_conditioning_scale": CONTROL_SCALE,
                "control_guidance_end": CONTROL_END,
            }
        torch = self._torch
        with self._lock:
            embeds, negative_embeds, pooled, negative_pooled = self._embed(prompt, negative)
            callback = None
            if strength_map is not None and mask.any():
                strength = float(np.clip(strength_map[mask].max(), 0.05, 1.0))
                callback = self._differential(image, strength_map, strength, seed)
            # The UNet nearly fills an 8 GB card. Windows usually spills past
            # that into system memory, but when RAM is short too an
            # allocation fails outright; freeing the allocator's cache and
            # trying once more gets through.
            for attempt in (1, 2):
                try:
                    result = pipe(
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
                        callback_on_step_end=callback,
                        callback_on_step_end_tensor_inputs=["latents"] if callback else None,
                        **control,
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

    def _differential(self, image: np.ndarray, strength_map: np.ndarray, strength: float, seed: int):
        """A step callback giving every pixel its own strength (differential
        diffusion, Levin & Fried 2023). Denoising starts at the map's highest
        strength; a pixel with a lower one is held to the original image,
        noised to the current level, until its own start step, so it keeps
        that much more of what was there. One pass, any mix of strengths:
        holes generated outright beside surfaces only touched up."""
        from PIL import Image

        torch = self._torch
        pipe = self._pipe
        size = image.shape[0]
        pixels = pipe.image_processor.preprocess(Image.fromarray(image), height=size, width=size)
        with torch.inference_mode():
            pixels = pixels.to(self.device, dtype=pipe.vae.dtype)
            original = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor
        generator = torch.Generator("cpu").manual_seed(seed + 1)
        noise = torch.randn(original.shape, generator=generator).to(self.device, dtype=original.dtype)
        latent_size = original.shape[-1]
        per_latent = cv2.resize(strength_map.astype(np.float32), (latent_size, latent_size), interpolation=cv2.INTER_AREA)
        # The step each pixel starts at, counted from the first step taken:
        # int(), as diffusers truncates the schedule for img2img.
        steps = self.steps
        start = int(steps * strength) - np.floor(steps * np.clip(per_latent, 0.0, 1.0)).astype(np.int64)
        start = torch.from_numpy(start).to(self.device)

        def on_step_end(pipeline, index, timestep, kwargs):
            latents = kwargs["latents"]
            frozen = start > index  # not yet started after this step
            if bool(frozen.any()) and index + 1 < pipeline.num_timesteps:
                # The latents now sit at the next timestep of the schedule.
                schedule = pipeline.scheduler.timesteps
                first = getattr(pipeline.scheduler, "begin_index", None) or 0
                following = schedule[min(first + index + 1, len(schedule) - 1)]
                level = pipeline.scheduler.add_noise(original, noise, following.reshape(1))
                kwargs["latents"] = torch.where(frozen[None, None], level.to(latents.dtype), latents)
            return kwargs

        return on_step_end

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


def control_cached() -> bool:
    """The optional depth guidance (SDXL_DEPTH_CONTROL) is downloaded."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return False
    return all(isinstance(try_to_load_from_cache(SDXL_DEPTH_CONTROL, name), str) for name in CONTROL_FILES)


def download_models() -> list[str]:
    from huggingface_hub import snapshot_download

    # One file at a time: parallel downloads into the cache race on Windows
    # when it cannot create symlinks.
    return [
        snapshot_download(SDXL_INPAINT, allow_patterns=list(SDXL_FILES), max_workers=1),
        snapshot_download(SDXL_VAE, allow_patterns=list(VAE_FILES), max_workers=1),
    ]


def download_control() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(SDXL_DEPTH_CONTROL, allow_patterns=list(CONTROL_FILES), max_workers=1)


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
#: Outdoors a horizon view keeps only what it paints up to this elevation;
#: the tilted views paint everything above. At the top of a frame the model
#: composes a photograph: a big out-of-focus branch hanging into it, a
#: blurred patch beside it. Looking up, the same trees are painted as
#: treetops against the sky.
HORIZON_TOP = 35.0
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
            if direction == "up" and not scene.enclosed and textured_fraction(image, known_view) >= CANOPY_FRACTION:
                # Trees (or anything) already rise into the view from below:
                # the open-sky prompt fought them into big blurred branches.
                direction = "canopy"
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
            top = HORIZON_TOP if direction == "around" and not scene.enclosed else None
            blend_view(panorama, known, protected, merged, soft, view, top_deg=top)
            done.append(view)
            logger.debug("view yaw %+.0f pitch %+.0f: %.0f%% generated", view.yaw_deg, view.pitch_deg, 100 * hole.mean())
        if progress:
            progress(index + 1, len(plan))
    if not known.all():
        # View borders the ring never revisited: a few thin slivers at most.
        panorama[:] = propagate_background(panorama, ~known, known)
    return Completion(panorama, ~original, done, time.perf_counter() - started)


#: An upward view whose known pixels are at least this textured (foliage,
#: branches, rock, buildings) is painted as a canopy, not as open sky.
CANOPY_FRACTION = 0.2
TEXTURE_STD = 8.0


def textured_fraction(image: np.ndarray, known: np.ndarray) -> float:
    """Share of the ``known`` pixels of a view with fine texture (local grey
    standard deviation over ``TEXTURE_STD``): foliage, branches, rock and
    buildings are textured, sky and cloud are smooth."""
    if known.sum() < 100:
        return 0.0
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    size = (9, 9)
    mean = cv2.blur(grey, size)
    std = np.sqrt(np.maximum(cv2.blur(grey * grey, size) - mean * mean, 0.0))
    return float((std[known] > TEXTURE_STD).mean())


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
