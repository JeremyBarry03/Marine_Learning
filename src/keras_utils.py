"""Shared TensorFlow/Keras utilities used across training and inference."""
from __future__ import annotations

from typing import Sequence

from tensorflow.keras.layers import Lambda as _LambdaLayer

_LAMBDA_PATCHED = False


def ensure_preprocess_lambda_deserializable(target_shape: Sequence[int]) -> None:
    """
    Patches Keras' Lambda deserialization so our preprocessing layer
    (named ``"preprocess"``) always reports an output shape.

    Newer Keras releases require Lambda layers loaded from disk to declare
    ``output_shape``; older checkpoints (and the ones we ship) omitted it,
    which raises ``NotImplementedError`` when loading the model. This helper
    amends the config at deserialization time without altering the saved file.
    """
    global _LAMBDA_PATCHED
    if _LAMBDA_PATCHED:
        return

    original_from_config = _LambdaLayer.from_config.__func__
    shape_list = list(target_shape)

    def _patched_from_config(cls, config):
        if config.get("name") == "preprocess" and not config.get("output_shape"):
            config["output_shape"] = shape_list
        return original_from_config(cls, config)

    _LambdaLayer.from_config = classmethod(_patched_from_config)
    _LAMBDA_PATCHED = True


def restore_preprocess_lambda(model, backbone: str) -> None:
    """
    Replaces the serialized lambda's closure so inference works under Keras 3.

    When older checkpoints are loaded, the `preprocess_fn` captured by the lambda
    arrives as a serialized dictionary rather than a callable. This helper swaps in
    the real preprocessing routine for the configured backbone.
    """
    try:
        layer = model.get_layer("preprocess")
    except ValueError:
        return

    if not hasattr(layer, "function"):
        return

    from src.model import BACKBONES  # imported lazily to avoid circular deps

    backbone_entry = BACKBONES.get(backbone)
    if not backbone_entry:
        raise ValueError(f"Unsupported backbone '{backbone}' while restoring preprocess lambda.")

    preprocess_fn = backbone_entry["preprocess"]

    def _patched_lambda(t, _fn=preprocess_fn):
        return _fn(t * 255.0)

    layer.function = _patched_lambda
    layer._function = _patched_lambda
