"""
Training entry point for the benthic species classifier.

This script performs a two-phase schedule:
1. Warm-up training with the ImageNet backbone frozen.
2. Optional fine-tuning of the upper backbone layers.

Artifacts produced:
- Trained model weights (`models/marine_classifier.h5`)
- Training metrics (`models/metrics/training_history.json`)
- Class name mapping (`models/metrics/class_names.json`)
- Dataset split manifest (`models/splits.json`) via `data_processing.build_datasets`
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import tensorflow as tf

from src.config import MARINE_CONFIG
from src.data_processing import build_datasets
from src.model import build_classifier, enable_fine_tuning


def _prepare_directories(config: Dict) -> None:
    root = Path(config["project_root"])
    training_cfg = config["training"]
    (root / training_cfg["checkpoint_path"]).parent.mkdir(parents=True, exist_ok=True)
    (root / training_cfg["history_path"]).parent.mkdir(parents=True, exist_ok=True)
    (root / training_cfg["class_names_path"]).parent.mkdir(parents=True, exist_ok=True)


def _configure_mixed_precision(config: Dict) -> None:
    if config["training"].get("mixed_precision", False):
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)


def _serialize_history(history_objects, output_path: Path) -> None:
    """Saves concatenated history objects to JSON for later inspection."""
    metrics: Dict[str, list] = {}
    for history in history_objects:
        for key, values in history.history.items():
            metrics.setdefault(key, []).extend(values)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def _save_class_names(class_names, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)


def main(config: Dict) -> None:
    _prepare_directories(config)
    _configure_mixed_precision(config)

    train_ds, val_ds, test_ds, class_names = build_datasets(config)
    _save_class_names(class_names, Path(config["project_root"]) / config["training"]["class_names_path"])

    model, base_model = build_classifier(config)

    training_cfg = config["training"]
    root = Path(config["project_root"])
    checkpoint_path = root / training_cfg["checkpoint_path"]

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    warmup_epochs = training_cfg["warmup_epochs"]
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    fine_tune_epochs = training_cfg["fine_tune_epochs"]
    if fine_tune_epochs > 0:
        enable_fine_tuning(base_model, config["model"]["fine_tune_from"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_cfg["fine_tune_learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False, label_smoothing=training_cfg.get("label_smoothing", 0.0)
            ),
            metrics=["accuracy"],
        )
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs + fine_tune_epochs,
            initial_epoch=warmup_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        history_objects = [history_warmup, history_finetune]
    else:
        history_objects = [history_warmup]

    best_model = tf.keras.models.load_model(checkpoint_path)
    train_metrics = best_model.evaluate(train_ds, verbose=0)
    val_metrics = best_model.evaluate(val_ds, verbose=0)
    test_metrics = best_model.evaluate(test_ds, verbose=0)

    history_path = root / training_cfg["history_path"]
    _serialize_history(history_objects, history_path)

    summary_path = root / training_cfg["training_summary_path"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    training_summary = {
        "train_loss": train_metrics[0],
        "train_accuracy": train_metrics[1],
        "val_loss": val_metrics[0],
        "val_accuracy": val_metrics[1],
        "test_loss": test_metrics[0],
        "test_accuracy": test_metrics[1],
        "warmup_epochs": warmup_epochs,
        "fine_tune_epochs": fine_tune_epochs,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2)

    print("Training complete. Evaluation summary:")
    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main(MARINE_CONFIG)
