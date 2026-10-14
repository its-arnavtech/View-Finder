"""Quality profiles: one knob that sizes every stage for the target device.

A phone and a desktop GPU can render very different meshes, and the depth
network's cost scales with pixel count, so the profile is chosen once and every
stage reads its budget from it. The ``mobile`` profile is sized so every mesh
primitive stays within 16-bit indices (<= 65,535 vertices), which halves index
buffer size and avoids relying on a WebGL1 extension for the common case.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QualityProfile:
    name: str
    # Long side the photo is resized to before depth estimation and meshing.
    working_long_side: int
    # Vertices along the long side of the (padded) mesh grid.
    grid_long: int
    # Long side of the embedded JPEG textures. <= 4096 keeps phones happy.
    texture_long_side: int
    jpeg_quality: int
    # Scene depth range in normalised units: the nearest pixel sits at 1.0.
    far: float
    # Fraction of the image size added on every side as a soft continuation.
    pad_fraction: float
    # Two neighbouring grid vertices whose disparity (1/z, near = 1) differs
    # by more than this belong to different surfaces; triangles spanning them
    # are cut. Disparity, not log depth: the gap a viewer sees opening when
    # they move is proportional to the disparity jump, and a smooth ground
    # plane has constant disparity gradient all the way to the horizon, so it
    # is never cut. Scales inversely with grid density.
    edge_disparity_jump: float
    # Dilation of the occluded region the background layer covers, as a
    # fraction of the working image's long side.
    occlusion_dilation_fraction: float
    sky_sphere_segments: int
    # Equirectangular surround texture (width, height) shown when the viewer
    # looks away from the photo. Power-of-two so WebGL1 phones can mipmap it.
    environment_size: tuple[int, int] = (1024, 512)


PROFILES: dict[str, QualityProfile] = {
    "mobile": QualityProfile(
        name="mobile",
        working_long_side=1024,
        grid_long=160,
        texture_long_side=1536,
        jpeg_quality=82,
        far=40.0,
        pad_fraction=0.12,
        edge_disparity_jump=0.05,
        occlusion_dilation_fraction=0.025,
        sky_sphere_segments=24,
    ),
    "desktop": QualityProfile(
        name="desktop",
        working_long_side=1536,
        grid_long=320,
        texture_long_side=2048,
        jpeg_quality=88,
        far=40.0,
        pad_fraction=0.12,
        edge_disparity_jump=0.03,
        occlusion_dilation_fraction=0.025,
        sky_sphere_segments=32,
        environment_size=(2048, 1024),
    ),
    "high": QualityProfile(
        name="high",
        working_long_side=2048,
        grid_long=512,
        texture_long_side=4096,
        jpeg_quality=92,
        far=40.0,
        pad_fraction=0.12,
        edge_disparity_jump=0.02,
        occlusion_dilation_fraction=0.025,
        sky_sphere_segments=48,
        environment_size=(2048, 1024),
    ),
}

DEFAULT_PROFILE = "desktop"


def get_profile(name: str | QualityProfile | None = None) -> QualityProfile:
    if isinstance(name, QualityProfile):
        return name
    key = (name or DEFAULT_PROFILE).strip().lower()
    try:
        return PROFILES[key]
    except KeyError:
        raise ValueError(
            f"Unknown quality profile {name!r}; choose one of {', '.join(PROFILES)}"
        ) from None


def profile_dict(profile: QualityProfile) -> dict[str, object]:
    return asdict(profile)
