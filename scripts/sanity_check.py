import json
from pathlib import Path
from collections import defaultdict

def load_json(file_path="00_data/01_raw/dataset_info.json"):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)

def print_header(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title) + "\n")

def sanity_check_dataset(dataset_information):
    stats = dataset_information["stats"]
    classes = dataset_information["classes"]

    print_header("Dataset Sanity Check")

    total_train = stats["total_train_images"]
    total_test = stats["total_test_images"]
    total_global = total_train + total_test

    print(f"{'Total training images':<23}: {total_train}")
    print(f"{'Total testing images':<23}: {total_test}")
    print(f"{'Total images in dataset':<23}: {total_global}\n")

    print_header("Class Distribution - Train")
    train_counts = {}
    for class_id, class_name in classes.items():
        counter = stats["train"][class_name]["count"]
        train_counts[class_name] = counter
        print(f"{class_name:<23}: {counter:6d} ({(counter/total_train)*100:5.2f}%)")

    print_header("Class Distribution - Test")
    test_counts = {}
    for class_id, class_name in classes.items():
        counter = stats["test"][class_name]["count"]
        test_counts[class_name] = counter
        print(f"{class_name:<23}: {counter:6d} ({(counter/total_test)*100:5.2f}%)")

if __name__ == "__main__":
    dataset_info = load_json()
    sanity_check_dataset(dataset_info)
    