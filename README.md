# Histology Classification Project

# Table of Contents

- [Project Structure](#project-structure)

## Project Structure

``` plaintext
Histology_Classification
├── 00_data
│   └── 01_raw
│       ├── test
│       │   ├── estroma
│       │   ├── inflamacion_benigna
│       │   └── tumores
│       ├── train
│       │   ├── estroma
│       │   ├── inflamacion_benigna
│       │   └── tumores
│       └── dataset_info.json
├── 10_preprocessing
│   └── 03_augmented
├── 20_modeling
├── 30_training
│   └── model_checkpoints
├── 40_evaluation
├── 50_results
│   ├── confusion_matrices
│   ├── grad_cam_images
│   └── training_curves
├── 60_documentation
│   ├── model_card.md
│   ├── project_specification.md
│   └── result_analysis.md
├── config
│   ├── config.yml
│   └── hyperparameters.yml
├── scripts
│   └── setup_data.py
├── README.md
├── Recommendation.txt
├── environment.yml
└── requirements.txt
```

Installing libraries after creating virtual environment:

``` powershell
pip install -r requirements.txt
```

