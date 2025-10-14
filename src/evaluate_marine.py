"""
Evaluation script for the benthic classifier.

Loads the persisted train/val/test split, evaluates the saved model on the
test partition, and stores detailed metrics plus a confusion-matrix plot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import MARINE_CONFIG
from src.data_processing import build_datasets
from src.keras_utils import ensure_preprocess_lambda_deserializable


def _collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Returns stacked arrays of true labels and predicted probabilities."""
    all_labels: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    for batch_images, batch_labels in dataset:
        logits = model.predict(batch_images, verbose=0)
        all_labels.append(batch_labels.numpy())
        all_logits.append(logits)
    return np.concatenate(all_labels, axis=0), np.concatenate(all_logits, axis=0)


def _compute_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    preds = np.argmax(predictions, axis=1)
    return tf.math.confusion_matrix(labels=labels, predictions=preds, num_classes=num_classes).numpy()


def _per_class_metrics(confusion_matrix: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for idx, class_name in enumerate(class_names):
        tp = float(confusion_matrix[idx, idx])
        fp = float(confusion_matrix[:, idx].sum() - tp)
        fn = float(confusion_matrix[idx, :].sum() - tp)
        support = float(confusion_matrix[idx, :].sum())

        precision = tp / (tp + fp + 1e-8) if support else 0.0
        recall = tp / (tp + fn + 1e-8) if support else 0.0
        f1 = (2 * precision * recall) / (precision + recall + 1e-8) if support else 0.0

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return metrics


def _plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(confusion_matrix.shape[1]),
        yticks=np.arange(confusion_matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = confusion_matrix.max() / 2.0 if confusion_matrix.size else 0.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j,
                i,
                format(confusion_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main(config: Dict) -> None:
    _, _, test_ds, class_names = build_datasets(config)
    model_path = Path(config["project_root"]) / config["training"]["checkpoint_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_marine.py first.")

    ensure_preprocess_lambda_deserializable(
        (config["data"]["image_size"], config["data"]["image_size"], 3)
    )
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    labels, predictions = _collect_predictions(model, test_ds)

    num_classes = len(class_names)
    confusion_matrix = _compute_confusion_matrix(labels, predictions, num_classes)
    per_class = _per_class_metrics(confusion_matrix, class_names)

    overall_accuracy = float(np.mean(np.argmax(predictions, axis=1) == labels))

    report = {
        "overall_accuracy": overall_accuracy,
        "per_class": per_class,
    }

    report_path = Path(config["project_root"]) / config["evaluation"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    cm_path = Path(config["project_root"]) / config["evaluation"]["confusion_matrix_path"]
    _plot_confusion_matrix(confusion_matrix, class_names, cm_path)

    print(f"Evaluation complete. Accuracy: {overall_accuracy:.4f}")
    print(f"Metrics written to {report_path}")
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main(MARINE_CONFIG)
