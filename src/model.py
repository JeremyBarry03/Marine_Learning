"""Model definitions for benthic species classification."""
from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


BACKBONES = {
    "EfficientNetB0": {
        "builder": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
    "EfficientNetB1": {
        "builder": tf.keras.applications.EfficientNetB1,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
    "ResNet50": {
        "builder": tf.keras.applications.ResNet50,
        "preprocess": tf.keras.applications.resnet.preprocess_input,
    },
}


def _build_augmentation(cfg: Dict) -> tf.keras.Sequential:
    """Creates a Sequential of augmentation layers driven by config."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip(cfg.get("flip", "horizontal"), name="aug_flip"),
            layers.RandomRotation(cfg.get("rotation", 0.1), name="aug_rotation"),
            layers.RandomZoom(cfg.get("zoom", 0.1), name="aug_zoom"),
            layers.RandomContrast(cfg.get("contrast", 0.1), name="aug_contrast"),
        ],
        name="data_augmentation",
    )


def build_classifier(config: Dict) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds and compiles a transfer learning model using the configured backbone.

    Returns `(model, base_model)` so callers can optionally fine-tune later.
    """
    model_cfg = config["model"]
    training_cfg = config["training"]
    image_size = config["data"]["image_size"]
    num_classes = len(config["data"]["species"])
    input_shape = (image_size, image_size, 3)

    if model_cfg["backbone"] not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {model_cfg['backbone']}")

    backbone_entry = BACKBONES[model_cfg["backbone"]]
    backbone_builder = backbone_entry["builder"]
    preprocess_fn = backbone_entry["preprocess"]
    base_model = backbone_builder(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = model_cfg.get("train_base", False)

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = _build_augmentation(model_cfg["augmentation"])(inputs)
    x = layers.Lambda(lambda t: preprocess_fn(t * 255.0), name="preprocess")(x)
    features = base_model(x, training=False)

    pooled = layers.GlobalAveragePooling2D(name="avg_pool")(features)
    dropped = layers.Dropout(model_cfg["dropout"], name="dropout")(pooled)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(dropped)

    model = models.Model(inputs=inputs, outputs=outputs, name="marine_classifier")
    return model, base_model


def enable_fine_tuning(base_model: tf.keras.Model, fine_tune_from: int) -> None:
    """
    Unfreezes layers from `fine_tune_from` onward on the provided backbone.

    A negative index will count from the end (e.g., -20 keeps all but the
    last 20 layers frozen).
    """
    base_model.trainable = True
    if fine_tune_from is None:
        return
    total_layers = len(base_model.layers)
    if fine_tune_from < 0:
        fine_tune_from = max(total_layers + fine_tune_from, 0)
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False


