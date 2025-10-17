"""
Utilities for running YOLO-based benthic detection.

The module lazily loads the `best.pt` checkpoint exported from the Colab
notebook and exposes a simple helper that returns bounding boxes in
pixel coordinates together with class labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from ultralytics import YOLO

# Class ordering taken from the README detection dataset table / YOLO manifest.
DETECTION_CLASSES: Sequence[str] = (
    "Crab",
    "Eel",
    "Flatfish",
    "Roundfish",
    "Scallop",
    "Skate",
    "Whelk",
)

_detector: YOLO | None = None
_loaded_checkpoint: Path | None = None


def _load_model(checkpoint_path: Path) -> YOLO:
    """
    Loads the YOLO checkpoint if needed. Reuses a singleton instance so we avoid
    reloading weights for every /detect request.
    """
    global _detector, _loaded_checkpoint
    checkpoint_path = checkpoint_path.resolve()
    if _detector is None or _loaded_checkpoint != checkpoint_path:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint missing at {checkpoint_path}")
        _detector = YOLO(str(checkpoint_path))
        _loaded_checkpoint = checkpoint_path
    return _detector


def _clip_box(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """
    Clamps bounding-box coordinates so we stay within the viewport.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), float(width)))
    x2 = max(0.0, min(float(x2), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    y2 = max(0.0, min(float(y2), float(height)))
    return x1, y1, x2, y2


def run_detection(
    image: Image.Image,
    checkpoint_path: Path,
    confidence_threshold: float = 0.25,
) -> Dict[str, object]:
    """
    Executes the YOLO detector on the provided PIL image and returns a summary
    dictionary containing bounding boxes and class labels.

    Bounding boxes are returned in absolute (x1, y1, x2, y2) pixel coordinates
    to simplify rendering on the front-end.
    """
    model = _load_model(checkpoint_path)
    width, height = image.size

    results = model.predict(
        image,
        conf=confidence_threshold,
        verbose=False,
    )
    if not results:
        return {
            "detections": [],
            "image_size": {"width": width, "height": height},
        }

    detections: List[Dict[str, object]] = []
    boxes = results[0].boxes

    if boxes is None or boxes.shape[0] == 0:
        return {
            "detections": [],
            "image_size": {"width": width, "height": height},
        }

    for i in range(boxes.shape[0]):
        cls_idx = int(boxes.cls[i].item())
        confidence = float(boxes.conf[i].item())
        cls_idx = max(0, min(cls_idx, len(DETECTION_CLASSES) - 1))
        label = DETECTION_CLASSES[cls_idx]

        # Extract bounding box in xyxy format (already absolute pixel coords)
        xyxy = boxes.xyxy[i].detach()
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu()
        x1, y1, x2, y2 = _clip_box(tuple(float(v) for v in xyxy.tolist()), width, height)

        detections.append(
            {
                "species": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }
        )

    return {
        "detections": detections,
        "image_size": {"width": width, "height": height},
    }


__all__ = ["run_detection", "DETECTION_CLASSES"]
