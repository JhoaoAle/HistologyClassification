# Histology Classification Project

# Table of Contents

- [Project Structure](#project-structure)

## Project Structure

``` plaintext
Histology_Classification
├── 00_data
│   ├── 01_raw
│   │   ├── test
│   │   │   ├── estroma
│   │   │   ├── inflamacion_benigna
│   │   │   └── tumores
│   │   ├── train
│   │   │   ├── estroma
│   │   │   ├── inflamacion_benigna
│   │   │   └── tumores
│   │   └── dataset_info.json
│   ├── 02_augmented
│   │   └── train
│   │       ├── estroma
│   │       └── inflamacion_benigna
│   └── 03_train_dataset
│           ├── estroma
│           ├── inflamacion_benigna
│           └── tumores
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
├── _training
│   └── model_checkpoints
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

Updating project structure tree by evaluating context on repo folder (Arch)

``` terminal
tree -I '.git|__pycache__|*.pyc|312_dl|.env|.ipynb_checkpoints|*.png' --dirsfirst
```

## Pipeline

### Downloading data
``` powershell
python scripts/setup_data.py
```

A file called dataset_info.json will be created, where information regarding the dataset can be read, to identify missing values or detect class imbalances. Optionally, the following script can be ran to see a friendlier report of the aforementioned file:

``` powershell
python scripts/sanity_check.py
```

``` powershell
python scripts/data_augmentation_torch.py
```

``` powershell
python scripts/merge_into_train.py
```


