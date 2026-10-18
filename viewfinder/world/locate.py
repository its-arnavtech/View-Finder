"""Where a photo was taken, and what kind of country surrounds it.

The generator paints whatever the prompt says wherever the photo gives it
little to go on: behind the camera, above, below. CLIP's place for a photo
can be a cliché of it: a lake among the low granite islands of Ontario was
"a mountain lake", and the world came back with alpine peaks and meadows
behind you. GeoCLIP, the project's geolocator (viewfinder/geolocator.py),
puts the photo on the globe first, and from there:

* the region's name (offline reverse geocoding) goes into the prompt;
* the kind of country there (``TERRAINS``) is scored against the location
  in GeoCLIP's image-location space, alongside the photo itself;
* of the places the photo plausibly shows, the likeliest there is chosen.

GeoCLIP's image tower is CLIP ViT-L/14, frozen: the same model the scene
classifier already holds, so only GeoCLIP's projection, its location encoder
(the project's fine-tuned one when present) and its GPS gallery are loaded
here, about 40 MB.

Trust: GeoCLIP's share of probability within ``REGION_KM`` of its best
guess. Measured on photos from eleven regions, every outdoor photo it placed
in the right region scored 0.65 or more, every misplaced one 0.44 or less;
below ``TRUST`` the location is not used at all. Rooms and vehicles never
use it: GeoCLIP knows countries, not interiors.

Everything runs locally; ``VIEWFINDER_WORLD_LOCATE=0`` turns it off.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

LOCATE_ENV = "VIEWFINDER_WORLD_LOCATE"
REGION_KM = 500.0
TRUST = 0.6
# Location evidence in CLIP-logit units per unit of cosine similarity, times
# the trust. A close call between two plausible places (the Ontario lake:
# 0.63 "mountain lake" against 0.27 "calm lake among forested hills") turns;
# a clear one (a sea coast in Hawaii, 0.79) does not.
LOCATION_WEIGHT = 25.0

# The kind of country around a natural scene: terrain and vegetation, for
# the unseen surroundings.
TERRAINS: tuple[str, ...] = (
    "boreal forest and lakes with granite outcrops, pine and spruce trees",
    "temperate deciduous forest with low rolling hills",
    "mixed conifer forest in rugged mountains",
    "high alpine mountains with snowy peaks, glaciers and meadows",
    "temperate rainforest with huge mossy conifers",
    "tropical rainforest with dense jungle and palms",
    "tropical island coast with palm trees and volcanic hills",
    "mediterranean hills with olive trees, scrub and limestone",
    "dry savanna grassland with scattered acacia trees",
    "hot desert with sand dunes",
    "arid red rock desert with canyons and mesas",
    "dry outback scrubland with red earth and eucalyptus trees",
    "eucalyptus forest with rocky sandstone gorges",
    "flat prairie grassland",
    "flat farmland with fields and hedgerows",
    "green rolling pastures with stone walls",
    "windswept moorland and heather on bare hills",
    "arctic tundra with bare rock and low shrubs",
    "rocky coastline with sea cliffs",
    "wetlands and marshes with reeds",
    "bamboo forest and terraced green hills",
    "volcanic landscape with black lava rock",
    "steppe with dry grass and distant mountains",
)


@dataclass(frozen=True)
class Location:
    lat: float
    lon: float
    confidence: float  # GeoCLIP's probability within REGION_KM of (lat, lon)
    region: str  # e.g. "Ontario, Canada"

    @property
    def trusted(self) -> bool:
        return self.confidence >= TRUST

    def as_dict(self) -> dict[str, object]:
        # A guess at where the photo was taken; one decimal (~10 km) is all
        # the prompts use.
        return {
            "lat": round(self.lat, 1),
            "lon": round(self.lon, 1),
            "confidence": round(self.confidence, 3),
            "region": self.region,
            "trusted": self.trusted,
        }


def enabled() -> bool:
    return os.environ.get(LOCATE_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def haversine_km(origin: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Great-circle distance from ``origin`` (lat, lon degrees) to each of
    ``points`` (N, 2)."""
    a, b = np.radians(np.asarray(origin, np.float64)), np.radians(np.asarray(points, np.float64))
    h = np.sin((b[:, 0] - a[0]) / 2) ** 2 + np.cos(a[0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[1]) / 2) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def regional_mass(probabilities: np.ndarray, coordinates: np.ndarray, radius_km: float = REGION_KM) -> tuple[int, float]:
    """GeoCLIP's best gallery entry and the share of its probability within
    ``radius_km`` of it. A low share means the guesses are spread over
    places far apart: GeoCLIP does not know where the photo is."""
    best = int(np.argmax(probabilities))
    near = haversine_km(coordinates[best], coordinates) <= radius_km
    return best, float(probabilities[near].sum() / max(float(probabilities.sum()), 1e-12))


def rerank(probabilities: np.ndarray, similarity: np.ndarray, confidence: float, allowed: np.ndarray,
           plausible: float = 0.1) -> int:
    """The likeliest choice given the photo (``probabilities``) and the
    location (cosine ``similarity`` of each choice to it, weighted by
    ``confidence``). Only choices the photo itself makes plausible (at least
    ``plausible`` times the best) and ``allowed`` ones compete: location
    breaks a close call, it never overrules what the photo shows."""
    best = int(np.argmax(probabilities))
    floor = plausible * float(probabilities[best])
    candidates = [i for i in range(len(probabilities)) if allowed[i] and probabilities[i] >= floor]
    if not candidates:
        return best
    score = np.log(np.maximum(probabilities, 1e-12)) + LOCATION_WEIGHT * confidence * similarity
    return max(candidates, key=lambda i: float(score[i]))


def region_name(lat: float, lon: float) -> str:
    """'Ontario, Canada', offline. The country's name needs Babel; without
    it the first-level region alone."""
    from viewfinder.reverse_geocode import reverse_geocode

    place = reverse_geocode(lat, lon)
    country = place.get("country", "")
    try:
        from babel import Locale

        country = Locale("en").territories.get(country, "") if country else ""
    except Exception:  # Babel missing: the region's own name still helps
        country = ""
    parts = [p for p in (place.get("admin1", ""), country) if p]
    return ", ".join(dict.fromkeys(parts))


class Locator:
    """GeoCLIP on a CLIP ViT-L/14 that is already loaded (the scene
    classifier's)."""

    def __init__(self, clip_model, processor, device: str) -> None:
        import torch
        from geoclip.model.location_encoder import LocationEncoder
        from geoclip.model.misc import file_dir, load_gps_data

        self._torch = torch
        self._clip, self._processor, self.device = clip_model, processor, device
        projection = torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.ReLU(), torch.nn.Linear(768, 512))
        projection.load_state_dict(torch.load(f"{file_dir}/weights/image_encoder_mlp_weights.pth", map_location="cpu"))
        self._projection = projection.to(device).eval()
        encoder = LocationEncoder()
        self.encoder_source = "geoclip-stock"
        try:
            from viewfinder.geolocator import _resolve_location_encoder

            state = torch.load(_resolve_location_encoder(None), map_location="cpu", weights_only=True)
            encoder.load_state_dict(state.get("location_encoder", state), strict=True)
            self.encoder_source = "fine-tuned"
        except FileNotFoundError:
            logger.info("No fine-tuned location encoder; using GeoCLIP's own")
        encoder = encoder.to(device).eval()
        self._scale = float(torch.load(f"{file_dir}/weights/logit_scale_weights.pth", map_location="cpu").exp())
        gallery = load_gps_data(f"{file_dir}/gps_gallery/coordinates_100K.csv").float()
        self._coordinates = gallery.numpy().astype(np.float64)
        with torch.no_grad():
            features = [
                torch.nn.functional.normalize(encoder(gallery[i:i + 20000].to(device)), dim=-1)
                for i in range(0, len(gallery), 20000)
            ]
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self._gallery = torch.cat(features).to(dtype)
        del encoder

    def text_features(self, texts: list[str]):
        """Texts in GeoCLIP's space: CLIP text features through the image
        projection (CLIP puts a photo and its description together)."""
        torch = self._torch
        inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            raw = self._clip.get_text_features(**inputs).float()
            return torch.nn.functional.normalize(self._projection(raw), dim=-1)

    def locate(self, image_features) -> tuple[Location, object]:
        """``image_features``: the photo's CLIP image features (projected,
        not normalised). Returns the location and its GeoCLIP feature (the
        probability-weighted mean of the gallery's, for scoring texts)."""
        torch = self._torch
        with torch.no_grad():
            image = torch.nn.functional.normalize(self._projection(image_features.float()), dim=-1)
            logits = self._scale * (image.to(self._gallery.dtype) @ self._gallery.T).float()
            probabilities = logits.softmax(dim=-1)[0]
            feature = torch.nn.functional.normalize(
                (probabilities[None].to(self._gallery.dtype) @ self._gallery).float(), dim=-1
            )
        best, confidence = regional_mass(probabilities.cpu().numpy().astype(np.float64), self._coordinates)
        lat, lon = (float(v) for v in self._coordinates[best])
        region = region_name(lat, lon) if confidence >= TRUST else ""
        return Location(lat, lon, confidence, region), feature

    @staticmethod
    def similarity(feature, texts) -> np.ndarray:
        return (feature @ texts.T)[0].float().cpu().numpy()
