"""
PyTorch inference helpers for the marine benthic classifier.

These utilities mirror the Colab training notebook so we can
load the exported DenseNet121 weights (`densenet121_benthic_final.pth`)
and perform inference from the Flask app.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# Default class ordering used during training on Colab.
DEFAULT_CLASS_NAMES: Sequence[str] = (
    "Scallop",
    "Roundfish",
    "Crab",
    "Whelk",
    "Skate",
    "Flatfish",
    "Eel",
)

# Image size and normalization stats copied from the notebook.
IMAGE_SIZE = 224
_EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(IMAGE_SIZE * 1.1)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _build_model(num_classes: int) -> nn.Module:
    """
    Reconstructs the DenseNet121 architecture with the same classifier head
    (Dropout + Linear) that was trained in Colab.
    """
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_class_names(class_names_path: Path | None) -> List[str]:
    """
    Loads class names from JSON if available, otherwise falls back
    to the default ordering used during training.
    """
    if class_names_path and class_names_path.exists():
        try:
            with class_names_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list) and all(isinstance(item, str) for item in loaded):
                return loaded
        except (OSError, json.JSONDecodeError):
            pass
    return list(DEFAULT_CLASS_NAMES)


class MarineClassifier:
    """
    Thin wrapper around the DenseNet121 model plus preprocessing pipeline.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        class_names_path: Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = load_class_names(class_names_path)
        self.model = _build_model(len(self.class_names))

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=self.device)
        # Accept either a pure state_dict or a dict with a 'model' key.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state and not isinstance(state["model"], int):
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> torch.Tensor:
        """
        Applies the evaluation transform, runs the model, and returns
        class probabilities as a 1D tensor on CPU.
        """
        tensor = _EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.squeeze(0).cpu()


__all__ = ["MarineClassifier", "load_class_names", "IMAGE_SIZE", "DEFAULT_CLASS_NAMES"]
