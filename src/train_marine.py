"""
Training entry point for the benthic species classifier.

This script performs a two-phase schedule:
1. Warm-up training with the ImageNet backbone frozen.
2. Optional fine-tuning of the upper backbone layers.

Artifacts produced:
- Trained model weights (`models/marine_classifier.keras`)
- Training metrics (`models/metrics/training_history.json`)
- Class name mapping (`models/metrics/class_names.json`)
- Accuracy curve plot (`models/metrics/accuracy_curve.png`)
- Dataset split manifest (`models/splits.json`) via `data_processing.build_datasets`
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable
import math

import tensorflow as tf

from src.config import MARINE_CONFIG
from src.data_processing import build_datasets
from src.keras_utils import ensure_preprocess_lambda_deserializable
from src.model import build_classifier, enable_fine_tuning


def _prepare_directories(config: Dict) -> None:
    root = Path(config["project_root"])
    training_cfg = config["training"]
    (root / training_cfg["checkpoint_path"]).parent.mkdir(parents=True, exist_ok=True)
    (root / training_cfg["history_path"]).parent.mkdir(parents=True, exist_ok=True)
    (root / training_cfg["class_names_path"]).parent.mkdir(parents=True, exist_ok=True)
    (root / training_cfg["accuracy_plot_path"]).parent.mkdir(parents=True, exist_ok=True)


def _configure_mixed_precision(config: Dict) -> None:
    if config["training"].get("mixed_precision", False):
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)


def _serialize_history(history_objects, output_path: Path) -> Dict[str, list]:
    """Saves concatenated history objects to JSON for later inspection."""
    metrics: Dict[str, list] = {}
    for history in history_objects:
        for key, values in history.history.items():
            metrics.setdefault(key, []).extend(values)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def _plot_accuracy(history: Dict[str, list], output_path: Path, warmup_epochs: int, fine_tune_epochs: int) -> None:
    """Generates an accuracy vs. epoch plot for training and validation."""
    train_accuracy = history.get("accuracy")
    val_accuracy = history.get("val_accuracy")

    if not train_accuracy or not val_accuracy:
        return

    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_accuracy) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Marine Classifier Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    if warmup_epochs and fine_tune_epochs:
        transition_epoch = min(len(epochs), warmup_epochs)
        if 0 < transition_epoch < len(epochs):
            plt.axvline(
                transition_epoch,
                color="tab:gray",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Fine-tune Start",
            )
            plt.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_class_names(class_names, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)


def _build_loss(training_cfg: Dict) -> tf.keras.losses.Loss:
    """Constructs the classification loss with safe fallback for smoothing."""
    label_smoothing = training_cfg.get("label_smoothing", 0.0)
    try:
        return tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            label_smoothing=label_smoothing,
        )
    except TypeError:
        if label_smoothing:
            tf.get_logger().warning(
                "SparseCategoricalCrossentropy does not support label_smoothing "
                "on this TensorFlow build. Falling back to 0.0."
            )
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def _build_optimizer(training_cfg: Dict, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    """Instantiates the configured optimizer for a given learning rate."""
    optimizer_name = training_cfg.get("optimizer", "adam").lower()
    weight_decay = training_cfg.get("weight_decay", 0.0)

    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=training_cfg.get("momentum", 0.9),
            nesterov=training_cfg.get("nesterov", False),
        )
    # Default to Adam for backward compatibility.
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


class CosineWithPlateauCallback(tf.keras.callbacks.Callback):
    """Epoch-wise cosine annealing schedule with optional minimum learning rate."""

    def __init__(
        self,
        base_lr: float,
        total_epochs: int,
        start_epoch: int = 0,
        min_lr_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_lr = base_lr
        self.total_epochs = max(total_epochs, 1)
        self.start_epoch = start_epoch
        self.min_lr_ratio = min_lr_ratio

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        if epoch < self.start_epoch:
            return
        relative_epoch = epoch - self.start_epoch
        relative_epoch = min(relative_epoch, self.total_epochs - 1)

        min_lr = self.base_lr * self.min_lr_ratio
        if self.total_epochs <= 1:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (self.base_lr - min_lr) * (
                1 + math.cos(math.pi * relative_epoch / max(self.total_epochs - 1, 1))
            )
        _set_optimizer_learning_rate(self.model.optimizer, lr)


def _set_optimizer_learning_rate(optimizer: tf.keras.optimizers.Optimizer, lr: float) -> None:
    """Robustly updates optimizer LR across TF versions."""
    if hasattr(optimizer, "learning_rate"):
        lr_attr = optimizer.learning_rate
    elif hasattr(optimizer, "lr"):
        lr_attr = optimizer.lr
    else:
        lr_attr = None

    if hasattr(lr_attr, "assign"):
        lr_attr.assign(lr)
    elif isinstance(lr_attr, tf.Variable):
        tf.keras.backend.set_value(lr_attr, lr)
    elif lr_attr is not None:
        try:
            optimizer.learning_rate = lr
        except (AttributeError, TypeError):
            optimizer._set_hyper("learning_rate", lr)
    else:
        optimizer._set_hyper("learning_rate", lr)


def _build_callbacks(
    training_cfg: Dict,
    checkpoint_cb: tf.keras.callbacks.Callback,
    base_lr: float,
    total_epochs: int,
    start_epoch: int = 0,
) -> Iterable[tf.keras.callbacks.Callback]:
    """Creates callbacks tailored to the configured training strategy."""
    callbacks = [checkpoint_cb]

    schedule_type = training_cfg.get("lr_schedule")
    if schedule_type == "cosine_with_plateau":
        callbacks.append(
            CosineWithPlateauCallback(
                base_lr=base_lr,
                total_epochs=max(total_epochs, 1),
                start_epoch=start_epoch,
                min_lr_ratio=training_cfg.get("lr_min_factor", 0.1),
            )
        )
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=base_lr * 1e-4,
                verbose=1,
            )
        )
    else:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            )
        )

    early_cfg = training_cfg.get("early_stopping")
    if early_cfg:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_cfg.get("monitor", "val_loss"),
                patience=early_cfg.get("patience", 5),
                restore_best_weights=True,
                verbose=1,
            )
        )

    return callbacks


def main(config: Dict) -> None:
    _prepare_directories(config)
    _configure_mixed_precision(config)

    train_ds, val_ds, test_ds, class_names = build_datasets(config)
    _save_class_names(class_names, Path(config["project_root"]) / config["training"]["class_names_path"])

    model, base_model = build_classifier(config)

    training_cfg = config["training"]
    root = Path(config["project_root"])
    checkpoint_path = root / training_cfg["checkpoint_path"]

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor=training_cfg.get("checkpoint_monitor", "val_accuracy"),
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    loss_fn = _build_loss(training_cfg)

    warmup_epochs = training_cfg["warmup_epochs"]
    warmup_lr = training_cfg["learning_rate"]
    model.compile(
        optimizer=_build_optimizer(training_cfg, warmup_lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    warmup_callbacks = _build_callbacks(
        training_cfg,
        checkpoint_cb,
        base_lr=warmup_lr,
        total_epochs=warmup_epochs,
        start_epoch=0,
    )
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=list(warmup_callbacks),
        verbose=1,
    )

    fine_tune_epochs = training_cfg["fine_tune_epochs"]
    if fine_tune_epochs > 0:
        enable_fine_tuning(base_model, config["model"]["fine_tune_from"])
        fine_tune_lr = training_cfg["fine_tune_learning_rate"]
        model.compile(
            optimizer=_build_optimizer(training_cfg, fine_tune_lr),
            loss=loss_fn,
            metrics=["accuracy"],
        )
        fine_tune_callbacks = _build_callbacks(
            training_cfg,
            checkpoint_cb,
            base_lr=fine_tune_lr,
            total_epochs=fine_tune_epochs,
            start_epoch=warmup_epochs,
        )
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs + fine_tune_epochs,
            initial_epoch=warmup_epochs,
            callbacks=list(fine_tune_callbacks),
            verbose=1,
        )
        history_objects = [history_warmup, history_finetune]
    else:
        history_objects = [history_warmup]

    ensure_preprocess_lambda_deserializable(
        (config["data"]["image_size"], config["data"]["image_size"], 3)
    )
    best_model = tf.keras.models.load_model(checkpoint_path, safe_mode=False)
    train_metrics = best_model.evaluate(train_ds, verbose=0)
    val_metrics = best_model.evaluate(val_ds, verbose=0)
    test_metrics = best_model.evaluate(test_ds, verbose=0)

    history_path = root / training_cfg["history_path"]
    history_metrics = _serialize_history(history_objects, history_path)

    accuracy_plot_path = root / training_cfg["accuracy_plot_path"]
    _plot_accuracy(history_metrics, accuracy_plot_path, warmup_epochs, fine_tune_epochs)

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
