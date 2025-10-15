"""Flask inference service for the marine classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from src.config import MARINE_CONFIG
from src.keras_utils import ensure_preprocess_lambda_deserializable, restore_preprocess_lambda

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "webapp" / "public"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)

CONFIG = MARINE_CONFIG
IMAGE_SIZE = CONFIG["data"]["image_size"]
MODEL_PATH = PROJECT_ROOT / CONFIG["training"]["checkpoint_path"]
CLASS_NAMES_PATH = PROJECT_ROOT / CONFIG["training"]["class_names_path"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. Run src/train_marine.py before starting the API."
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file missing at {CLASS_NAMES_PATH}. Ensure train_marine.py completed successfully."
    )

with CLASS_NAMES_PATH.open("r", encoding="utf-8") as handle:
    CLASS_NAMES: List[str] = json.load(handle)

ensure_preprocess_lambda_deserializable((IMAGE_SIZE, IMAGE_SIZE, 3))
MODEL = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
restore_preprocess_lambda(MODEL, CONFIG["model"]["backbone"])

SPECIES_FACTS: Dict[str, Dict[str, str]] = {
    "Scallop": {"summary": "Bivalve mollusk often found on sandy seafloor habitats."},
    "Roundfish": {"summary": "General term for oval-bodied demersal fish species."},
    "Crab": {"summary": "Decapod crustaceans adapted to benthic crawling and scavenging."},
    "Whelk": {"summary": "Predatory sea snails known for spiral shells on soft substrates."},
    "Skate": {"summary": "Cartilaginous fish with diamond-shaped bodies and wing-like fins."},
    "Flatfish": {"summary": "Bottom-dwelling fish with asymmetric bodies camouflaged against seabed."},
    "Eel": {"summary": "Elongated fish species occupying crevices and soft-bottom habitats."},
}


def _preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image).astype("float32") / 255.0
    return array


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file_storage = request.files["file"]
    if file_storage.filename == "":
        return jsonify({"error": "Empty filename supplied."}), 400

    image = Image.open(file_storage.stream)
    processed = _preprocess_image(image)
    batch = np.expand_dims(processed, axis=0)

    logits = MODEL.predict(batch, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()
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


if __name__ == "__main__":
    app.run(debug=True)
