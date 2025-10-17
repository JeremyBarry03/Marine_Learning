"""Flask inference service for the marine classifier and YOLO detector."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from src.detection_service import run_detection
from src.pytorch_inference import MarineClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "webapp" / "public"
CLASSIFICATION_CHECKPOINT = PROJECT_ROOT / "densenet121_benthic_final.pth"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "metrics" / "class_names.json"
DETECTION_CHECKPOINT = PROJECT_ROOT / "marine_yolo_v8n_best" / "best.pt"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)


def _load_image_from_request() -> Tuple[Image.Image, str]:
    if "file" not in request.files:
        raise ValueError("No file uploaded.")

    file_storage = request.files["file"]
    if file_storage.filename == "":
        raise ValueError("Empty filename supplied.")

    try:
        image = Image.open(file_storage.stream)
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    return image.convert("RGB"), file_storage.filename


if not CLASSIFICATION_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"PyTorch checkpoint not found at {CLASSIFICATION_CHECKPOINT}. "
        "Ensure densenet121_benthic_final.pth has been exported from training."
    )

CLASSIFIER = MarineClassifier(
    checkpoint_path=CLASSIFICATION_CHECKPOINT,
    class_names_path=CLASS_NAMES_PATH if CLASS_NAMES_PATH.exists() else None,
)
CLASS_NAMES: List[str] = CLASSIFIER.class_names

SPECIES_FACTS: Dict[str, Dict[str, str]] = {
    "Scallop": {"summary": "Bivalve mollusk often found on sandy seafloor habitats."},
    "Roundfish": {"summary": "General term for oval-bodied demersal fish species."},
    "Crab": {"summary": "Decapod crustaceans adapted to benthic crawling and scavenging."},
    "Whelk": {"summary": "Predatory sea snails known for spiral shells on soft substrates."},
    "Skate": {"summary": "Cartilaginous fish with diamond-shaped bodies and wing-like fins."},
    "Flatfish": {"summary": "Bottom-dwelling fish with asymmetric bodies camouflaged against seabed."},
    "Eel": {"summary": "Elongated fish species occupying crevices and soft-bottom habitats."},
}


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image, _ = _load_image_from_request()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    probabilities = CLASSIFIER.predict(image).numpy()
    top_index = int(np.argmax(probabilities))
    top_species = CLASS_NAMES[top_index]
    confidence = float(probabilities[top_index])

    top_k = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {"species": CLASS_NAMES[idx], "confidence": float(probabilities[idx])}
        for idx in top_k
    ]

    response = {
        "primary_prediction": {
            "species": top_species,
            "confidence": confidence,
            "description": SPECIES_FACTS.get(top_species, {}).get("summary", ""),
        },
        "top_predictions": top_predictions,
    }
    return jsonify(response)


@app.route("/detect", methods=["POST"])
def detect():
    if not DETECTION_CHECKPOINT.exists():
        return (
            jsonify(
                {
                    "error": "Detection checkpoint is unavailable on the server.",
                }
            ),
            503,
        )

    try:
        image, _ = _load_image_from_request()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    summary = run_detection(image, DETECTION_CHECKPOINT)
    return jsonify(summary)


if __name__ == "__main__":
    app.run(debug=True)
