"""
Project-wide configuration for marine benthic classification.
Using a Python dictionary keeps configuration flexible while
remaining easy to import across training, evaluation, and inference.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MARINE_CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "data": {
        "image_dir": "data/data/classification_dataset/images",
        "labels_path": "data/data/classification_dataset/labels.txt",
        "split_manifest_path": "models/splits.json",
        "species": [
            "Scallop",
            "Roundfish",
            "Crab",
            "Whelk",
            "Skate",
            "Flatfish",
            "Eel",
        ],
        "image_size": 224,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
        "seed": 415,
    },
    "model": {
        "backbone": "ResNet50",
        "dropout": 0.3,
        "train_base": False,
        "fine_tune_from": -20,  # unfreeze last 20 layers after warmup
        "augmentation": {
            "flip": "horizontal_and_vertical",
            "rotation": 0.1,
            "zoom": 0.1,
            "contrast": 0.1,
        },
    },
    "training": {
        "batch_size": 32,
        "warmup_epochs": 10,
        "fine_tune_epochs": 20,
        "learning_rate": 1e-4,
        "fine_tune_learning_rate": 5e-5,
        "label_smoothing": 0.0,
        "mixed_precision": False,
        "checkpoint_path": "models/marine_classifier.h5",
        "history_path": "models/metrics/training_history.json",
        "class_names_path": "models/metrics/class_names.json",
        "training_summary_path": "models/metrics/training_summary.json",
    },
    "evaluation": {
        "report_path": "models/metrics/evaluation.json",
        "confusion_matrix_path": "models/metrics/confusion_matrix.png",
    },
}
