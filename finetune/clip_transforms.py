"""Preprocessing transforms matching GeoCLIP's *inference* path.

geoclip ships two inconsistent preprocessing routes:

* `geoclip.train.dataloader.img_train_transform` / `img_val_transform` normalize
  with ImageNet statistics (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225).
* `GeoCLIP.predict()` preprocesses via `ImageEncoder.preprocess_image`, i.e. the
  HuggingFace CLIP processor, which normalizes with CLIP statistics
  (0.48145466, 0.4578275, 0.40821073) / (0.26862954, 0.26130258, 0.27577711).

The pretrained weights are deployed through the second route, so evaluating the
pretrained baseline with ImageNet stats would feed the frozen tower
out-of-distribution inputs, understate the baseline, and inflate any measured
"improvement" from fine-tuning. We therefore use CLIP statistics everywhere, and
match the processor's resize/crop geometry (bicubic shortest-edge 224 + centre
crop 224) for validation.

Named clip_transforms rather than transforms so it cannot shadow
`torchvision.transforms` for anything that imports by bare module name.
"""

from __future__ import annotations

import torch
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

BICUBIC = transforms.InterpolationMode.BICUBIC


def clip_val_transform():
    """Deterministic preprocessing equivalent to the CLIP processor's."""
    return transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def clip_train_transform():
    """geoclip's training augmentation, but with CLIP normalization statistics."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
