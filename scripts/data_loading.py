import yaml
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import torch
import numpy as np


# --------------------------------------------------
# Load config
# --------------------------------------------------
def load_config(path="config/config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# Create weights for class balancing
# --------------------------------------------------
def make_balanced_sampler(dataset, full_dataset):
    targets = [full_dataset.samples[i][1] for i in dataset.indices]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


# --------------------------------------------------
# Main loader builder
# --------------------------------------------------
def load_dataloaders():
    cfg = load_config()

    paths = cfg["paths"]["data"]
    data_cfg = cfg["data"]
    aug_cfg  = cfg["augmentation"]

    raw_root   = Path(paths["raw"])
    train_root = raw_root / "train"
    test_root  = raw_root / "test"

    image_size = tuple(data_cfg["image_size"])
    batch_size = data_cfg["batch_size"]
    val_split  = data_cfg.get("validation_split", 0.2)
    use_balancing = data_cfg.get("use_balancing", True)

    # --------------------------------------------------
    # Transforms
    # --------------------------------------------------
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(aug_cfg["rotation_range"]),
        transforms.RandomHorizontalFlip(p=0.5 if aug_cfg["horizontal_flip"] else 0.0),
        transforms.RandomVerticalFlip(p=0.5 if aug_cfg["vertical_flip"] else 0.0),
        transforms.ColorJitter(
            brightness=aug_cfg["brightness_range"],
            contrast=aug_cfg["contrast_range"],
            saturation=aug_cfg["saturation_range"],
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(aug_cfg["width_shift_range"], aug_cfg["height_shift_range"]),
            scale=(1 - aug_cfg["zoom_range"], 1 + aug_cfg["zoom_range"]),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------
    full_train_dataset = datasets.ImageFolder(str(train_root), transform=train_transforms)
    test_dataset       = datasets.ImageFolder(str(test_root),  transform=test_transforms)

    num_classes = len(full_train_dataset.classes)

    # --------------------------------------------------
    # Train/Val split
    # --------------------------------------------------
    val_size   = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # --------------------------------------------------
    # Loaders
    # --------------------------------------------------
    if use_balancing:
        train_sampler = make_balanced_sampler(train_dataset, full_train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)

    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # --------------------------------------------------
    # Diagnostics (prints once)
    # --------------------------------------------------

    print("DATASET SUMMARY")
    print(f" Train samples:      {len(train_dataset)}")
    print(f" Validation samples: {len(val_dataset)}")
    print(f" Test samples:       {len(test_dataset)}")
    print(f" Classes:            {full_train_dataset.classes}")
    print(f" Class balancing:    {use_balancing}")
    print("---------------------------------------------\n")

    return train_loader, val_loader, test_loader, num_classes, full_train_dataset.classes
