import shutil
from pathlib import Path
import yaml

def load_config(path="config/config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()

    # Paths from config
    raw_train = Path(cfg["paths"]["data"]["raw"]) / "train"
    aug_train = Path(cfg["paths"]["data"]["augmented"]) / "train"

    # Create final merged dataset folder
    final_train = Path("00_data/03_train_dataset")
    final_train.mkdir(parents=True, exist_ok=True)

    # Class names from config
    class_names = cfg["data"]["classes"]

    print("\nMerging training sets:")
    print(" RAW:", raw_train)
    print(" AUG:", aug_train)
    print(" OUT:", final_train, "\n")

    for cls in class_names:
        cls_raw = raw_train / cls
        cls_aug = aug_train / cls
        cls_final = final_train / cls

        cls_final.mkdir(parents=True, exist_ok=True)

        # Copy originals
        if cls_raw.exists():
            for f in cls_raw.glob("*.png"):
                shutil.copy2(f, cls_final / f.name)

        # Copy augmented (only exists for minority classes)
        if cls_aug.exists():
            for f in cls_aug.glob("*.png"):
                shutil.copy2(f, cls_final / f.name)

        print(f" - {cls}: {len(list(cls_final.glob('*.png')))} files")

    print("\nMerged dataset created at:", final_train)

if __name__ == "__main__":
    main()