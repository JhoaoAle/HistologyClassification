import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

from data_loading import load_dataloaders
from model_cnn import build_model_from_config


# ------------------------------------------
# Load config
# ------------------------------------------
def load_config(path="config/config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main ():
    train_loader, val_loader, test_loader, num_classes, class_names = load_dataloaders()
    cfg = load_config()

    # Create result dirs
    results_cfg = cfg["paths"]["results"]

    curve_dir = Path(results_cfg["training_curves"])
    report_dir = Path(results_cfg["performance_reports"])
    cm_dir = Path(results_cfg["confusion_matrices"])
    model_out_dir = Path(cfg["paths"]["models"]["final_models"])

    for d in [curve_dir, report_dir, cm_dir, model_out_dir]:
        d.mkdir(parents=True, exist_ok=True)


    # ------------------------------------------
    # Build model
    # ------------------------------------------
    model, device = build_model_from_config(cfg)

    # ------------------------------------------
    # Training hyperparameters
    # ------------------------------------------
    epochs = cfg["training"]["epochs"]
    learning_rate = cfg["training"]["initial_learning_rate"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = Path(cfg["paths"]["models"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    train_accs = []


    # ------------------------------------------
    # Training Loop
    # ------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = train_loss / total
        epoch_acc = correct / total

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # Save checkpoint every epoch
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch}.pth")


    print("\nTraining complete.\n")


    # ==========================================
    # TEST EVALUATION
    # ==========================================
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    # ------------------------------------------
    # Save Classification Report
    # ------------------------------------------
    report = classification_report(
    all_labels, all_preds,
    target_names=class_names,
    output_dict=True
    )

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")

    with open(report_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved classification_report.json")


    # ------------------------------------------
    # Save Confusion Matrix
    # ------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_dir / "confusion_matrix.png")
    plt.close()

    print("Saved confusion_matrix.png")


    # ------------------------------------------
    # Save Training Curves
    # ------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(train_accs, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_dir / "training_curves.png")
    plt.close()

    with open(curve_dir / "training_curves.json", "w") as f:
        json.dump({"loss": train_losses, "accuracy": train_accs}, f, indent=2)

    print("Saved training_curves.* files")


    # ------------------------------------------
    # Save final model
    # ------------------------------------------
    torch.save(model.state_dict(), model_out_dir / "final_model.pth")
    print(f"Saved final_model.pth to {model_out_dir}")


if __name__ == "__main__":
    main()