"""GeoCLIP model construction and the freeze contract for location-encoder-only fine-tuning.

Freezing notes, verified against geoclip's source:

* `ImageEncoder.__init__` already sets requires_grad=False on every CLIP
  parameter, but it does NOT freeze `ImageEncoder.mlp` (the 768->768->512
  projection head). That head is part of the pretrained image-side weights
  (image_encoder_mlp_weights.pth), so leaving it trainable would mean we are
  fine-tuning the image side too. We freeze it explicitly.

* `logit_scale` is a bare nn.Parameter on GeoCLIP and is trainable by default.
  The brief is "location encoder only", so we freeze it as well. This also keeps
  the before/after comparison clean: the similarity temperature is identical
  between the pretrained and fine-tuned models, so any measured change comes
  from the location encoder alone.

* `LocationEncoder` (geoclip.model.location_encoder.LocationEncoder) is the MLP
  stack that maps GPS -> the shared 512-d embedding space. It is three parallel
  "capsules", each a random-Fourier-feature encoding at a different sigma
  (2**0, 2**4, 2**8) followed by a 3-layer MLP and a 512-d head; their outputs
  are summed. This is the only module we train.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from geoclip import GeoCLIP


def build_model(device: str = "cuda", queue_size: int = 1024, drop_text_tower: bool = True) -> GeoCLIP:
    """Load pretrained GeoCLIP onto `device`.

    `drop_text_tower` removes CLIP's text encoder, which `get_image_features`
    never touches. It is pure resident VRAM otherwise.
    """
    model = GeoCLIP(from_pretrained=True, queue_size=queue_size)

    if drop_text_tower:
        clip = model.image_encoder.CLIP
        # Replace with tiny stand-ins so any accidental use fails loudly rather
        # than silently producing garbage.
        if hasattr(clip, "text_model"):
            clip.text_model = nn.Identity()
        if hasattr(clip, "text_projection"):
            clip.text_projection = nn.Identity()

    model = model.to(device)
    return model


def freeze_image_side(model: GeoCLIP) -> None:
    """Freeze everything except the location encoder."""
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    model.logit_scale.requires_grad = False

    for param in model.location_encoder.parameters():
        param.requires_grad = True

    # The frozen image tower must not accumulate batchnorm/dropout state either.
    model.image_encoder.eval()


def assert_freeze_contract(model: GeoCLIP) -> None:
    """Hard check that only location encoder params are trainable."""
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    unexpected = {n for n in trainable if not n.startswith("location_encoder.")}
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters outside location encoder: {sorted(unexpected)}")
    if not trainable:
        raise RuntimeError("No trainable parameters -- location encoder was frozen by mistake.")

    loc_total = sum(1 for _, _ in model.location_encoder.named_parameters())
    loc_trainable = sum(1 for p in model.location_encoder.parameters() if p.requires_grad)
    if loc_total != loc_trainable:
        raise RuntimeError(f"Only {loc_trainable}/{loc_total} location encoder tensors are trainable.")


def trainable_parameter_report(model: GeoCLIP) -> str:
    def count(module: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, train

    img_total, img_train = count(model.image_encoder)
    mlp_total, mlp_train = count(model.image_encoder.mlp)
    loc_total, loc_train = count(model.location_encoder)
    scale_train = int(model.logit_scale.requires_grad)

    all_total = sum(p.numel() for p in model.parameters())
    all_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return (
        "parameter freeze report\n"
        f"  image_encoder (incl. CLIP) : {img_total:>12,} total | {img_train:>10,} trainable\n"
        f"    of which projection mlp  : {mlp_total:>12,} total | {mlp_train:>10,} trainable\n"
        f"  location_encoder           : {loc_total:>12,} total | {loc_train:>10,} trainable\n"
        f"  logit_scale                : {1:>12,} total | {scale_train:>10,} trainable\n"
        f"  ---------------------------------------------------------------\n"
        f"  MODEL TOTAL                : {all_total:>12,} total | {all_train:>10,} trainable "
        f"({100.0 * all_train / all_total:.3f}%)"
    )
