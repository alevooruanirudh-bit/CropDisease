"""Train/eval transforms.

The paper's augmentation is Keras ``ImageDataGenerator(rotation_range=40, zoom_range=0.115,
shear_range=0.115, horizontal_flip, vertical_flip, brightness_range=[0.5, 1.5],
fill_mode='nearest')``. Mapped to albumentations:

* rotation_range=40         -> Affine(rotate=(-40, 40))
* zoom_range=0.115          -> Affine(scale=(0.885, 1.115))
* shear_range=0.115         -> Affine(shear=(-0.115, 0.115)). Keras gives shear in *degrees*,
                               so this is a near-no-op; we keep it for fidelity.
* brightness_range=[.5,1.5] -> ColorJitter(brightness=(0.5, 1.5)) (multiplicative, like Keras)
* fill_mode='nearest'       -> border_mode=cv2.BORDER_REPLICATE
"""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(aug_cfg: dict, img_size: int) -> A.Compose:
    """Training transform from the ``augment`` config block."""
    t: list = []
    if aug_cfg.get("random_resized_crop", False):
        t.append(A.RandomResizedCrop(size=(img_size, img_size), scale=tuple(aug_cfg.get("rrc_scale", [0.7, 1.0])),
                                     ratio=(0.85, 1.18)))
    else:
        t.append(A.Resize(img_size, img_size))
    if aug_cfg.get("paper_geometric", True):
        t += [
            A.Affine(rotate=(-40, 40), scale=(0.885, 1.115), shear=(-0.115, 0.115),
                     border_mode=cv2.BORDER_REPLICATE, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=(0.5, 1.5), contrast=0, saturation=0, hue=0, p=1.0),
        ]
    cj = aug_cfg.get("color_jitter", 0.0)
    if cj:
        t.append(A.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(0.05, cj / 4), p=0.8))
    t += [A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()]
    return A.Compose(t)


def build_eval_transform(img_size: int) -> A.Compose:
    """Deterministic eval transform: resize + normalise."""
    return A.Compose([A.Resize(img_size, img_size), A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])


def build_transforms(cfg: dict) -> tuple[A.Compose, A.Compose]:
    """(train, eval) transforms. ``augment.enabled: false`` gives train = eval (ablation)."""
    s = cfg["model"]["img_size"]
    aug = cfg.get("augment", {})
    train = build_train_transform(aug, s) if aug.get("enabled", True) else build_eval_transform(s)
    return train, build_eval_transform(s)
