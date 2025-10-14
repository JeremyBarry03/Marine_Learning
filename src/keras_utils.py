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
