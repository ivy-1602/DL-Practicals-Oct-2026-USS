"""
augment_data.py
----------------
Expands the small seed set of Taylor Swift album covers (1-3 images per
palette class) into a larger synthetic dataset using augmentation
(rotation, flip, brightness/contrast jitter, zoom/crop).

Why: with only 12 source images total across 7 classes, a neural net
cannot learn anything meaningful from raw counts. Augmentation lets us
demonstrate the MLP pipeline (as required by the prac) on a dataset of
usable size. This is a standard technique for small image datasets.
"""

import os
from PIL import Image, ImageEnhance
import numpy as np

RAW_DIR = "data/organized"
OUT_DIR = "data/augmented"
IMG_SIZE = (64, 64)          # keep small -> flattened MLP input stays manageable
SAMPLES_PER_SOURCE = 40      # augmented copies generated per seed image

rng = np.random.default_rng(42)


def augment_image(img: Image.Image) -> Image.Image:
    """Apply a random combination of augmentations to one PIL image."""
    out = img.copy()

    # random rotation
    angle = rng.uniform(-15, 15)
    out = out.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    # random horizontal flip
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)

    # random brightness
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.75, 1.25))

    # random contrast
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.8, 1.2))

    # random color/saturation jitter
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.7, 1.3))

    # random zoom/crop
    w, h = out.size
    zoom = rng.uniform(0.85, 1.0)
    nw, nh = int(w * zoom), int(h * zoom)
    left = rng.integers(0, w - nw + 1)
    top = rng.integers(0, h - nh + 1)
    out = out.crop((left, top, left + nw, top + nh)).resize((w, h))

    return out


def main():
    classes = sorted(os.listdir(RAW_DIR))
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = []
    for cls in classes:
        cls_in = os.path.join(RAW_DIR, cls)
        cls_out = os.path.join(OUT_DIR, cls)
        os.makedirs(cls_out, exist_ok=True)

        sources = [f for f in os.listdir(cls_in) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        count = 0
        for src_name in sources:
            src_path = os.path.join(cls_in, src_name)
            base_img = Image.open(src_path).convert("RGB").resize(IMG_SIZE)

            # save the original (resized) too
            base_img.save(os.path.join(cls_out, f"orig_{src_name}"))
            count += 1

            for i in range(SAMPLES_PER_SOURCE):
                aug = augment_image(base_img)
                fname = f"{os.path.splitext(src_name)[0]}_aug{i}.jpg"
                aug.save(os.path.join(cls_out, fname))
                count += 1

        manifest.append((cls, count))
        print(f"{cls:15s}: {count} images (from {len(sources)} source covers)")

    total = sum(c for _, c in manifest)
    print(f"\nTotal augmented dataset size: {total} images across {len(classes)} classes")


if __name__ == "__main__":
    main()
