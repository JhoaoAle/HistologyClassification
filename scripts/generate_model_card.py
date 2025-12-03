import json
import yaml
from pathlib import Path
from datetime import datetime


# Load config file
def load_config(path="config/config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classification_report(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Utilities: robust path resolution for config entries
def resolve_path_entry(entry, default: Path):
    if isinstance(entry, (str, Path)):
        return Path(entry)
    if isinstance(entry, dict):
        # common candidate keys in config maps
        for key in ("base", "root", "path", "dir", "folder"):
            v = entry.get(key)
            if isinstance(v, (str, Path)):
                return Path(v)
        # fallback: first string value in dict
        for v in entry.values():
            if isinstance(v, (str, Path)):
                return Path(v)
    return Path(default)


def find_classification_report(start_dirs):
    filename = "classification_report.json"
    for d in start_dirs:
        p = Path(d) / "performance_reports" / filename
        if p.exists():
            return p

    # fallback: try direct values if start_dirs contain nested dict values (handled before)
    # last resort: search workspace for the file
    repo_root = Path.cwd()
    matches = list(repo_root.rglob(filename))
    return matches[0] if matches else None

def format_metrics_table(report):
    """Create markdown table for class-level metrics."""
    md = "| Class | Precision | Recall | F1-Score | Support |\n"
    md += "|-------|-----------|--------|----------|---------|\n"

    skip_keys = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    for cls in sorted(report.keys()):
        if cls in skip_keys:
            continue
        metrics = report.get(cls, {})
        try:
            prec = float(metrics.get("precision", 0.0))
            rec = float(metrics.get("recall", 0.0))
            f1 = float(metrics.get("f1-score", 0.0))
            sup = int(metrics.get("support", 0))
        except Exception:
            prec = rec = f1 = 0.0
            sup = 0
        md += f"| {cls} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {sup} |\n"

    return md


def format_summary_metrics(report):
    acc = float(report.get("accuracy", 0.0))
    f1_macro = float(report.get("macro avg", {}).get("f1-score", 0.0))
    f1_weighted = float(report.get("weighted avg", {}).get("f1-score", 0.0))

    return (
        "| Metric | Score |\n"
        "|--------|--------|\n"
        f"| **Accuracy** | {acc:.4f} |\n"
        f"| **Macro F1** | {f1_macro:.4f} |\n"
        f"| **Weighted F1** | {f1_weighted:.4f} |\n"
    )


# --------------------------------------------------------
# BUILD MODEL CARD CONTENT
# --------------------------------------------------------
def build_model_card(config, report):
    model_cfg = config.get("model", {}).get("cnn_from_scratch", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    results_paths = config.get("paths", {}).get("results", {})

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    classes_list = data_cfg.get("classes", [])
    classes_md = "\n".join(f"- {c}" for c in classes_list) if classes_list else "Not specified"

    return f"""
# Model Card — Cancer Tissue Classification
Generated: {now}

---

## Model Overview
This model performs 3-class classification of histology tissue images:

{classes_md}

The model is a custom CNN trained from scratch using PyTorch with on-the-fly data augmentation (when enabled in training configuration).

---

## Architecture Details (CNN from Scratch)

### Convolutional Blocks
- Filters: {model_cfg.get('filters', 'N/A')}
- Activation: ReLU
- Normalization: BatchNorm2d
- Downsampling: MaxPool2D
- Input shape: {config.get('model', {}).get('input_shape', 'N/A')}

### Dense Layers
- Fully connected layers: {model_cfg.get('dense_units', 'N/A')}
- Dropout rate: {model_cfg.get('dropout_rate', 'N/A')}
- Output classes: {config.get('model', {}).get('num_classes', 'N/A')}

---

## Dataset Information

- Image size: {data_cfg.get('image_size', ['N/A','N/A'])[0]}×{data_cfg.get('image_size', ['N/A','N/A'])[1]}
- Batch size: {data_cfg.get('batch_size', 'N/A')}
- Validation split: {data_cfg.get('validation_split', 'N/A')}
- Class balancing enabled: {data_cfg.get('use_balancing', False)}

Classes used:

{classes_md}

---

## Training Configuration

| Setting | Value |
|--------|--------|
| Epochs | {training_cfg.get('epochs', 'N/A')} |
| Learning Rate | {training_cfg.get('initial_learning_rate', 'N/A')} |
| Early Stopping Patience | {training_cfg.get('early_stopping_patience', 'N/A')} |
| LR Reduce Patience | {training_cfg.get('reduce_lr_patience', 'N/A')} |
| Optimizer | {training_cfg.get('optimizer', 'N/A')} |
| Loss Function | {training_cfg.get('loss_function', 'N/A')} |

---

## Evaluation Results

### Overall Metrics
{format_summary_metrics(report)}

---

### Per-Class Metrics
{format_metrics_table(report)}

---

## Confusion Matrix
Stored at:

`{resolve_path_entry(results_paths, Path('results')) / 'confusion_matrices'}/confusion_matrix.png`

---

## Grad-CAM
Enabled: {config.get('project', {}).get('requirements', {}).get('use_gradcam', False)}

Grad-CAM output folder:

`{resolve_path_entry(results_paths, Path('results')) / 'gradcam'}`

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
"""


def main():
    cfg = load_config()

    # robustly resolve results directory (the config may store it as string or dict)
    results_entry = cfg.get("paths", {}).get("results", {})
    results_dir = resolve_path_entry(results_entry, Path("results"))

    # try to find the classification report in likely places, fall back to repo search
    report_path = find_classification_report([results_dir])
    if report_path is None:
        # try if config contains a nested performance_reports entry
        if isinstance(results_entry, dict) and "performance_reports" in results_entry:
            report_path = resolve_path_entry(results_entry["performance_reports"], results_dir / "performance_reports") / "classification_report.json"

    if report_path is None or not Path(report_path).exists():
        # final attempt: check top-level paths.* entries for strings that might contain the file
        extra_candidates = []
        for v in cfg.get("paths", {}).values():
            if isinstance(v, (str, Path)):
                extra_candidates.append(Path(v))
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, (str, Path)):
                        extra_candidates.append(Path(vv))
        report_path = find_classification_report(extra_candidates)

    if not report_path or not Path(report_path).exists():
        raise FileNotFoundError("Could not find classification_report.json. Looked under config paths and repository.")

    report = load_classification_report(report_path)

    # resolve output path for model card (models.final_models may be string or dict)
    models_entry = cfg.get("paths", {}).get("models", {})
    final_models_entry = models_entry.get("final_models", "results/models")
    output_dir = resolve_path_entry(final_models_entry, Path("results/models"))
    output_path = Path(output_dir) / "model_card.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    card_text = build_model_card(cfg, report)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_text)

    print("\nModel card generated at:")
    print(output_path, "\n")


if __name__ == "__main__":
    main()