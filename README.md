# Marine Learning Platform

This repository hosts the end‑to‑end pipeline for classifying benthic organisms from underwater imagery. The project currently targets single-organism image classification across seven species (Scallop, Roundfish, Crab, Whelk, Skate, Flatfish, Eel) with a roadmap for future object detection.

## Repository Layout

- `data/data/classification_dataset/` — source images and `labels.txt` manifest provided by the challenge.
- `models/` — trained model checkpoints and training/evaluation metrics.
- `src/` — Python source code for data processing, model construction, training, evaluation, and the Flask inference service.
- `webapp/` — placeholder front-end artifacts (to be updated for marine theming later).

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training Workflow

1. **Configure:** Settings live in `src/config.py` (`MARINE_CONFIG`). It uses a 70/20/10 train/val/test split saved to `models/splits.json` so future runs reuse the exact same partition.
2. **Train:**  
   ```bash
   python -m src.train_marine
   ```
   - Performs a warm-up stage with the ResNet50 backbone frozen.
   - Optionally fine-tunes the upper backbone layers using a reduced learning rate.
   - Outputs:
     - `models/marine_classifier.h5` — ready-to-serve Keras model.
     - `models/metrics/class_names.json`
     - `models/metrics/training_history.json`
     - `models/metrics/training_summary.json`
     - `models/splits.json` (deterministic dataset split record)

3. **Evaluate:**  
   ```bash
   python -m src.evaluate_marine
   ```
   - Loads the saved model and test split.
   - Produces per-species precision/recall/F1 in `models/metrics/evaluation.json`.
   - Generates `models/metrics/confusion_matrix.png`.

## Running the Inference API

```bash
python -m src.app
```

- Accepts POST requests at `/predict` with an image file (`multipart/form-data`).
- Returns the primary species prediction, confidence score, and top-three ranked species.
- Uses metadata from `src/api_integration.py` (currently static placeholders) for quick species summaries.

Use `src/test_flask.py` as a simple smoke test once the server is running.

## Next Steps

1. **Hyperparameter search:** Explore additional backbones (EfficientNet variants, ConvNeXt) and fine-tune schedules for higher accuracy.
2. **Object detection module:** Extend the config and codebase with a `src/detection/` package to leverage the supplementary multi-organism dataset.
3. **Model explainability:** Add Grad-CAM or attention maps to aid marine scientists in interpreting classifications.
4. **Web experience:** Refresh `webapp/` to align with marine branding and surface model metrics/visualizations.
