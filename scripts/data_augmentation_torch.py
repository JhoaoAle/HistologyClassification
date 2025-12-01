import argparse
import json
import random
import shutil
import uuid
from pathlib import Path
from xml.etree.ElementPath import ops

import yaml
from PIL import Image
import torchvision.transforms as T


# ===============================================================
# LOAD CONFIG / JSON
# ===============================================================
def load_config(path="config/config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_info(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===============================================================
# BUILD TRANSFORM (FROM YOUR CONFIG)
# ===============================================================
def build_transform(cfg):
    ops = []

    # rotation_range
    if cfg.get("rotation_range", 0):
        ops.append(T.RandomRotation(cfg["rotation_range"]))

    # width/height shift using affine
    if cfg.get("width_shift_range") or cfg.get("height_shift_range"):
        ws = cfg.get("width_shift_range", 0.0)
        hs = cfg.get("height_shift_range", 0.0)
        ops.append(T.RandomAffine(degrees=0, translate=(ws, hs)))

    # zoom_range
    if cfg.get("zoom_range", 0):
        zr = cfg["zoom_range"]
        image_size = tuple(cfg.get("image_size", [224, 224]))
        ops.append(T.RandomResizedCrop(size=image_size, scale=(1 - zr, 1.0)))

    # flips
    if cfg.get("horizontal_flip", False):
        ops.append(T.RandomHorizontalFlip(p=0.5))
    if cfg.get("vertical_flip", False):
        ops.append(T.RandomVerticalFlip(p=0.5))

    # brightness / contrast / saturation
    ops.append(
        T.ColorJitter(
            brightness=cfg.get("brightness_range", [1, 1])[1] - 1,
            contrast=cfg.get("contrast_range", [1, 1])[1] - 1,
            saturation=cfg.get("saturation_range", [1, 1])[1] - 1,
        )
    )

    return T.Compose(ops)


# ===============================================================
# ALWAYS SAVE PNG
# ===============================================================
def save_png(img, out_path):
    out_path = out_path.with_suffix(".png")
    img.save(out_path, format="PNG")


# ===============================================================
# UPSAMPLE A SINGLE CLASS
# ===============================================================
def upsample_class(class_dir, target_count, out_dir, transform):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy originals first
    for f in class_dir.glob("*.png"):
        shutil.copy2(f, out_dir / f.name)

    # Determine how many extra images needed
    originals = list(class_dir.glob("*.png"))
    current = len(list(out_dir.glob("*.png")))
    need = target_count - current

    print(f"Upsampling class '{class_dir.name}': need {need} extra images.")

    i = 0
    while need > 0:
        src = originals[i % len(originals)]
        try:
            with Image.open(src) as img:
                w, h = img.size
                aug = transform(img)
                aug = aug.resize((w, h), Image.BILINEAR)

                new_name = f"{src.stem}_aug_{uuid.uuid4().hex[:8]}.png"
                save_png(aug, out_dir / new_name)
                need -= 1

        except Exception as e:
            print(f"Skipping {src}: {e}")

        i += 1


# ===============================================================
# MAIN
# ===============================================================
def main():
    parser = argparse.ArgumentParser(description="Augment minority classes only.")
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument("--json", default="00_data/01_raw/dataset_info.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load config + dataset_info
    cfg = load_config(args.config)
    dataset_info = load_dataset_info(args.json)

    raw_root = Path(cfg["paths"]["data"]["raw"])
    aug_root = Path(cfg["paths"]["data"]["augmented"])
    aug_root.mkdir(parents=True, exist_ok=True)

    aug_cfg = cfg.get("augmentation", {})
    transform = build_transform(aug_cfg)

    # Extract counts from JSON
    train_stats = dataset_info["stats"]["train"]

    class_counts = {cls: train_stats[cls]["count"] for cls in train_stats}
    print("Training class counts:", class_counts)

    max_count = max(class_counts.values())

    # Identify minority classes
    minority_classes = [cls for cls, c in class_counts.items() if c < max_count]

    print("\nMinority classes:", minority_classes)

    # Perform augmentation ONLY on minority classes
    for cls in minority_classes:
        class_dir = raw_root / "train" / cls
        out_dir = aug_root / "train" / cls

        print(f"\n--- Augmenting minority class: {cls} ---")
        upsample_class(class_dir, max_count, out_dir, transform)

    print("\nAll minority classes upsampled successfully.")
    print("Augmented dataset stored in:", aug_root)


if __name__ == "__main__":
    main()