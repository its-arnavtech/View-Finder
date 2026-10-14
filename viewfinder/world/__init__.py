"""World Engine: build an explorable 3D world from a single photo.

See ``docs/3d-engine/ARCHITECTURE.md``. Heavy imports (torch, transformers)
happen only when a depth model is actually loaded.
"""

SCHEMA = "viewfinder.world/1"

from .profiles import PROFILES, QualityProfile, get_profile  # noqa: E402
from .pipeline import WorldResult, build_world  # noqa: E402

__all__ = ["SCHEMA", "PROFILES", "QualityProfile", "WorldResult", "build_world", "get_profile"]
