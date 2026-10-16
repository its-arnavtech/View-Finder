"""What kind of place a photo shows, for prompting the surround generator.

The generator sees the photo itself through every view it fills, so the
prompt only has to keep it on the right track where it has little context
(behind the camera, straight up, straight down). CLIP (ViT-L/14, MIT) picks
the closest place, lighting and setting from fixed vocabularies, zero-shot.
Nothing here is tuned to a particular photo.

Generated surroundings are always *empty*: a room is continued as an empty
room and a square as an empty square, never with invented people.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

CLIP_MODEL = "openai/clip-vit-large-patch14"

# (phrase, setting). Settings: "nature" and "urban" are outdoors, "indoor"
# and "vehicle" are enclosed.
PLACES: tuple[tuple[str, str], ...] = (
    ("a mountain lake with a rocky shore", "nature"),
    ("a calm lake surrounded by forested hills", "nature"),
    ("a river flowing through a forest", "nature"),
    ("a mountain stream with boulders", "nature"),
    ("a waterfall in a gorge", "nature"),
    ("a dense forest", "nature"),
    ("a forest trail", "nature"),
    ("a tropical jungle", "nature"),
    ("a narrow slot canyon with a river", "nature"),
    ("a wide desert canyon", "nature"),
    ("a rocky desert", "nature"),
    ("sand dunes", "nature"),
    ("a sandy beach and the sea", "nature"),
    ("a tropical beach with palm trees", "nature"),
    ("a rocky sea coast with cliffs", "nature"),
    ("the open ocean", "nature"),
    ("snowy mountain peaks", "nature"),
    ("an alpine meadow below mountains", "nature"),
    ("rolling green hills", "nature"),
    ("a grassy field", "nature"),
    ("farmland and fields", "nature"),
    ("a savanna", "nature"),
    ("a wetland marsh", "nature"),
    ("a glacier and ice", "nature"),
    ("a volcanic landscape", "nature"),
    ("a cave with rock formations", "nature"),
    ("a garden with plants and paths", "urban"),
    ("a city park with trees and paths", "urban"),
    ("a city street lined with buildings", "urban"),
    ("a narrow old town street", "urban"),
    ("a public square in a city", "urban"),
    ("a city skyline across water", "urban"),
    ("a quiet residential street with houses", "urban"),
    ("a harbor with boats", "urban"),
    ("a bridge over a river", "urban"),
    ("a temple courtyard", "urban"),
    ("the outside of a church or cathedral", "urban"),
    ("a castle", "urban"),
    ("ancient stone ruins", "urban"),
    ("a road through the countryside", "urban"),
    ("a highway", "urban"),
    ("a train station platform", "urban"),
    ("a parking lot", "urban"),
    ("a stadium", "urban"),
    ("a living room", "indoor"),
    ("a bedroom", "indoor"),
    ("a kitchen", "indoor"),
    ("a bathroom", "indoor"),
    ("an office", "indoor"),
    ("a classroom", "indoor"),
    ("a library", "indoor"),
    ("a museum gallery", "indoor"),
    ("the inside of a church", "indoor"),
    ("a restaurant dining room", "indoor"),
    ("a cafe", "indoor"),
    ("a shop", "indoor"),
    ("a shopping mall", "indoor"),
    ("an airport terminal", "indoor"),
    ("a hotel lobby", "indoor"),
    ("a corridor", "indoor"),
    ("a gym", "indoor"),
    ("a concert hall", "indoor"),
    ("a warehouse", "indoor"),
    ("a subway station", "indoor"),
    ("an indoor swimming pool", "indoor"),
    ("the cabin of an airplane", "vehicle"),
    ("the inside of a train carriage", "vehicle"),
    ("the inside of a car", "vehicle"),
    ("the inside of a bus", "vehicle"),
)

LIGHTING: tuple[str, ...] = (
    "on a sunny day",
    "on an overcast day",
    "at sunset",
    "at sunrise",
    "at night",
    "in fog",
    "lit by indoor lights",
    "lit by daylight through windows",
)

# Outdoors only. Without it the generator happily puts snow-capped peaks
# beside an autumn lake.
SEASONS: tuple[str, ...] = (
    "in spring",
    "in summer",
    "in autumn with autumn foliage",
    "in winter with snow",
)

# Built places only (urban, indoor, vehicles): what they are made of. "A
# corridor" alone came back as a modern hotel hallway beside a Gothic stone
# one; the photo's own look has to be in the prompt where the model repaints
# a whole view.
STYLES: tuple[str, ...] = (
    "with gothic stone arches and vaulted ceilings",
    "with classical ornate plaster mouldings",
    "with marble floors and columns",
    "with dark wood panelling",
    "with white walls and light wooden floors",
    "with modern minimalist white walls",
    "with exposed brick walls",
    "with rustic timber beams",
    "with concrete and glass",
    "with industrial metal and concrete",
    "with colourful tiles",
    "with traditional japanese wooden architecture",
    "with old stone buildings",
    "with modern glass towers",
    "with painted wooden houses",
    "with asphalt and grass verges",
)

# Outdoors only: where the photo was taken from. From a height (a hillside,
# a tower, an aircraft) what lies below the horizon can be far below, and the
# ground prior (nothing below the horizon beyond a level ground at eye
# height) would flatten a valley into a plain at your feet.
VIEWPOINTS: tuple[tuple[str, bool], ...] = (
    ("a photo taken standing on the ground", False),
    ("a photo taken at eye level on a street, path or shore", False),
    ("a photo taken from a high viewpoint looking down over a valley far below", True),
    ("a photo taken from the top of a tall building looking down over a city", True),
    ("an aerial photo taken from an airplane window", True),
)

NEGATIVE = (
    "people, person, crowd, human, man, woman, child, figure, face, animal, "
    "text, letters, watermark, logo, signature, frame, border, collage, split screen, "
    "blurry, out of focus, motion blur, smeared, streaks, bokeh, depth of field, "
    "distorted, deformed, lowres, jpeg artifacts, cartoon, painting, illustration"
)


@dataclass(frozen=True)
class Scene:
    place: str
    setting: str  # nature | urban | indoor | vehicle
    lighting: str
    confidence: float
    season: str = ""  # outdoors only, e.g. "in autumn with autumn foliage"
    style: str = ""  # built places only, e.g. "with gothic stone arches and vaulted ceilings"
    elevated: bool = False  # outdoors, seen from well above the ground below the horizon

    @property
    def enclosed(self) -> bool:
        return self.setting in ("indoor", "vehicle")

    @property
    def subject(self) -> str:
        return f"{self.place} {self.style}".strip()

    @property
    def conditions(self) -> str:
        return f"{self.lighting} {self.season}".strip()

    def prompt(self, direction: str = "around") -> str:
        """Prompt for one generated view. ``direction`` is ``around`` (the
        horizon), ``up`` or ``down``."""
        if direction == "up":
            if self.enclosed:
                return f"looking up at the ceiling of {self.subject} {self.conditions}, photograph, empty, no people, detailed"
            return (
                f"looking up at the open sky above {self.subject} {self.conditions}, clouds, photograph, "
                "empty, no people, detailed"
            )
        if direction == "down":
            subject = "the floor" if self.enclosed else "the ground underfoot"
            return f"{subject} of {self.subject} {self.conditions}, photograph, empty, no people, detailed"
        empty = "an empty room, " if self.enclosed else "empty, "
        return (
            f"a wide-angle photograph of {self.subject} {self.conditions}, {empty}no people, "
            "natural colours, sharp focus, highly detailed, realistic"
        )

    def negative(self, direction: str = "around") -> str:
        """Negative prompt: never people or text; never haze unless the
        photo itself is foggy (generated haze builds up view after view).
        Looking straight up outdoors, nothing but sky: given trees at the
        edges of its view, the model otherwise grows an upside-down cliff
        across the zenith."""
        negative = NEGATIVE
        if self.lighting != "in fog":
            negative += ", haze, fog, mist, smoke, washed out, low contrast"
        if direction == "up" and not self.enclosed:
            negative += ", trees, branches, leaves, mountains, cliff, rocks, ground, land, water, buildings"
        return negative

    def as_dict(self) -> dict[str, object]:
        return {
            "place": self.place,
            "setting": self.setting,
            "lighting": self.lighting,
            "season": self.season,
            "style": self.style,
            "elevated": self.elevated,
            "confidence": round(self.confidence, 3),
            "prompt": self.prompt(),
        }


GENERIC = Scene("a natural landscape", "nature", "on a sunny day", 0.0)


class SceneClassifier:
    def __init__(self, device: str = "auto") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._torch = torch
        self._model = CLIPModel.from_pretrained(CLIP_MODEL, torch_dtype=dtype, local_files_only=True).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
        self._places = self._encode_text([f"a photo of {phrase}" for phrase, _ in PLACES])
        self._lighting = self._encode_text([f"a photo taken {phrase}" for phrase in LIGHTING])
        self._seasons = self._encode_text([f"a landscape photo taken {phrase}" for phrase in SEASONS])
        self._styles = self._encode_text([f"a photo of a place {phrase}" for phrase in STYLES])
        self._viewpoints = self._encode_text([phrase for phrase, _ in VIEWPOINTS])
        self._lock = threading.Lock()

    def _encode_text(self, texts: list[str]):
        torch = self._torch
        inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def classify(self, rgb: np.ndarray) -> Scene:
        torch = self._torch
        with self._lock, torch.no_grad():
            inputs = self._processor(images=rgb, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
            image = torch.nn.functional.normalize(self._model.get_image_features(**inputs).float(), dim=-1)
            place_p = (100.0 * image @ self._places.T).softmax(dim=-1)[0].cpu().numpy()
            light_p = (100.0 * image @ self._lighting.T).softmax(dim=-1)[0].cpu().numpy()
            season_p = (100.0 * image @ self._seasons.T).softmax(dim=-1)[0].cpu().numpy()
            style_p = (100.0 * image @ self._styles.T).softmax(dim=-1)[0].cpu().numpy()
            viewpoint_p = (100.0 * image @ self._viewpoints.T).softmax(dim=-1)[0].cpu().numpy()
        best = int(np.argmax(place_p))
        phrase, setting = PLACES[best]
        lighting = LIGHTING[int(np.argmax(light_p))]
        enclosed = setting in ("indoor", "vehicle")
        # An outdoor photo is never "lit by indoor lights", and vice versa.
        if enclosed and not lighting.startswith("lit by"):
            lighting = "lit by daylight through windows" if light_p[LIGHTING.index("lit by daylight through windows")] >= light_p[LIGHTING.index("lit by indoor lights")] else "lit by indoor lights"
        elif not enclosed and lighting.startswith("lit by"):
            outdoor = [i for i, text in enumerate(LIGHTING) if not text.startswith("lit by")]
            lighting = LIGHTING[max(outdoor, key=lambda i: light_p[i])]
        season = "" if enclosed else SEASONS[int(np.argmax(season_p))]
        style = STYLES[int(np.argmax(style_p))] if setting != "nature" else ""
        high = sum(float(p) for p, (_, up) in zip(viewpoint_p, VIEWPOINTS) if up)
        elevated = not enclosed and high > 0.5
        return Scene(phrase, setting, lighting, float(place_p[best]), season, style, elevated)


_cache: dict[str, SceneClassifier] = {}
_cache_lock = threading.Lock()


def load_scene_classifier(device: str = "auto") -> SceneClassifier:
    with _cache_lock:
        if device not in _cache:
            _cache[device] = SceneClassifier(device)
        return _cache[device]


def download_model() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        CLIP_MODEL,
        allow_patterns=["*.json", "*.txt", "model.safetensors"],
        max_workers=1,
    )


def clip_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return False
    return isinstance(try_to_load_from_cache(CLIP_MODEL, "config.json"), str)
