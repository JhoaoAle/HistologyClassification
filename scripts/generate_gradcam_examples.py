import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import yaml
import cv2

from data_loading import load_dataloaders
from model_cnn import build_model_from_config


# ------------------------------------------------------------
# Load config
# ------------------------------------------------------------
def load_config(path="config/config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------
# GradCAM Hook Class
# ------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):

        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, class_idx):

        # Global-average pooling of gradients
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])

        # Create CAM on the SAME device as activations
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32,
                        device=self.activations.device)

        # Weight channels
        for i, w in enumerate(pooled_grads):
            cam += w * self.activations[0, i, :, :]

        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.detach().cpu().numpy()

# ------------------------------------------------------------
# Visualization helper
# ------------------------------------------------------------
def overlay_gradcam(img, cam, alpha=0.4):
    """
    img: numpy image (H,W,3) in [0,1]
    cam: CAM heatmap (h,w) in [0,1]
    """

    # Resize CAM to image resolution
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    heatmap = heatmap[:, :, ::-1] / 255.0  # BGR → RGB & normalize

    # Overlay
    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)

    return overlay


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------
def main():

    cfg = load_config()

    # Output directory
    out_dir = Path(cfg["paths"]["results"]["gradcam"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, device = build_model_from_config(cfg)
    model.load_state_dict(torch.load(
        "50_results/trained_models/final_model.pth",
        map_location=device
    ))
    model.eval()

    # GRADCAM target layer (last conv layer)
    target_layer = model.features[-3]   # This is ALWAYS the last Conv2D in your SimpleCNN
    gradcam = GradCAM(model, target_layer)

    train_loader, val_loader, test_loader, num_classes, class_names = load_dataloaders()
    classes = test_loader.dataset.classes
    print("\nClasses:", classes)

    # Pick 12 random samples
    indices = random.sample(range(len(test_loader.dataset)), 12)

    # Load all images from dataset, not the loader
    base_dataset = test_loader.dataset

    print("\nGenerating Grad-CAM...")

    for idx in indices:

        img_tensor, label = base_dataset[idx]
        img = img_tensor.unsqueeze(0).to(device)

        # Forward pass
        output = model(img)
        pred_class = output.argmax(dim=1).item()

        # Backprop for predicted class
        model.zero_grad()
        output[0, pred_class].backward()

        # Generate Grad-CAM
        cam = gradcam.generate(pred_class)

        # Convert tensor → visualizable image
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        overlay = overlay_gradcam(img_np, cam)

        # Save result
        save_path = out_dir / f"sample_{idx}_pred_{classes[pred_class]}_true_{classes[label]}.png"

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"Original\nTrue: {classes[label]}")
        plt.imshow(img_np)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Grad-CAM\nPredicted: {classes[pred_class]}")
        plt.imshow(overlay)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    print("\n✔ Done! 12 Grad-CAM images saved to:", out_dir)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
