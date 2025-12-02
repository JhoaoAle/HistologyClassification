
# Model Card — Cancer Tissue Classification
Generated: 2025-12-02 16:18:48

---

## Model Overview
This model performs 3-class classification of histology tissue images:

- estroma
- inflamacion_benigna
- tumores

The model is a custom CNN trained from scratch using PyTorch with on-the-fly data augmentation (when enabled in training configuration).

---

## Architecture Details (CNN from Scratch)

### Convolutional Blocks
- Filters: [32, 64, 128, 256, 512]
- Activation: ReLU
- Normalization: BatchNorm2d
- Downsampling: MaxPool2D
- Input shape: [224, 224, 3]

### Dense Layers
- Fully connected layers: [1024, 256]
- Dropout rate: 0.2
- Output classes: 3

---

## Dataset Information

- Image size: 224×224
- Batch size: 32
- Validation split: 0.2
- Class balancing enabled: True

Classes used:

- estroma
- inflamacion_benigna
- tumores

---

## Training Configuration

| Setting | Value |
|--------|--------|
| Epochs | 100 |
| Learning Rate | 0.001 |
| Early Stopping Patience | 10 |
| LR Reduce Patience | 5 |
| Optimizer | adam |
| Loss Function | cross_entropy |

---

## Evaluation Results

### Overall Metrics
| Metric | Score |
|--------|--------|
| **Accuracy** | 0.8673 |
| **Macro F1** | 0.8536 |
| **Weighted F1** | 0.8689 |


---

### Per-Class Metrics
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| estroma | 0.756 | 0.818 | 0.786 | 1196 |
| inflamacion_benigna | 0.817 | 0.929 | 0.869 | 476 |
| tumores | 0.935 | 0.879 | 0.906 | 2692 |


---

## Confusion Matrix
Stored at:

`50_results\training_curves\confusion_matrices/confusion_matrix.png`

---

## Grad-CAM
Enabled: True

Grad-CAM output folder:

`50_results\training_curves\gradcam`

---

## Limitations
- Model trained only on the provided tissue classes.
- Not medically validated.
- Sensitive to staining variability.
- Inputs must match the model's expected patch size and channels.

---

## Intended Use
Research and educational use only.
Not intended for clinical diagnosis.
