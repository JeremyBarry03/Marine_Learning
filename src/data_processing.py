"""
Utilities for building TensorFlow datasets tailored to the benthic
classification task. The module reads the provided labels manifest,
creates stratified train/validation/test splits (70/20/10), and
emits preprocessed `tf.data.Dataset` instances ready for model
training and evaluation.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class DatasetSplit:
    """Holds filepaths and numeric labels for one dataset split."""

    filepaths: List[str]
    labels: List[int]


def _load_manifest(
    manifest_path: Path,
    image_dir: Path,
    class_to_index: Dict[str, int],
) -> List[Tuple[str, int]]:
    """Returns list of (filepath, label_index) pairs from the manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Labels manifest not found: {manifest_path}")

    records: List[Tuple[str, int]] = []
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            filename, label_name = parts[0], " ".join(parts[1:])
            label_key = label_name.lower()
            if label_key not in class_to_index:
                raise ValueError(f"Unknown class '{label_name}' in manifest.")
            image_path = image_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image referenced in manifest: {image_path}")
            records.append((str(image_path), class_to_index[label_key]))
    if not records:
        raise ValueError("No records found in manifest; verify dataset integrity.")
    return records


def _stratified_split(
    records: Sequence[Tuple[str, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Performs stratified splitting by label index."""
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for path, label in records:
        by_label.setdefault(label, []).append((path, label))

    rng = random.Random(seed)
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for label, items in by_label.items():
        rng.shuffle(items)
        total = len(items)
        train_count = max(1, round(total * train_ratio))
        val_count = max(1, round(total * val_ratio))
        if train_count + val_count >= total:
            # Ensure at least one example remains for the test split.
            val_count = max(1, total - train_count - 1)
        train_items = items[:train_count]
        val_items = items[train_count:train_count + val_count]
        test_items = items[train_count + val_count:]

        for path, lbl in train_items:
            train_paths.append(path)
            train_labels.append(lbl)
        for path, lbl in val_items:
            val_paths.append(path)
            val_labels.append(lbl)
        for path, lbl in test_items:
            test_paths.append(path)
            test_labels.append(lbl)

    return (
        DatasetSplit(train_paths, train_labels),
        DatasetSplit(val_paths, val_labels),
        DatasetSplit(test_paths, test_labels),
    )


def _save_split_manifest(split_path: Path, splits: Dict[str, DatasetSplit]) -> None:
    """Persists the filenames chosen for each split to disk."""
    split_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        name: [Path(path).name for path in split.filepaths]
        for name, split in splits.items()
    }
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def _load_split_manifest(split_path: Path) -> Dict[str, List[str]]:
    """Loads a previously saved split file."""
    with split_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {key: list(value) for key, value in data.items()}


def _make_tf_dataset(
    split: DatasetSplit,
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Converts a DatasetSplit into a batched `tf.data.Dataset`."""
    if not split.filepaths:
        raise ValueError("Received empty split; ensure split ratios are correct.")

    paths_tensor = tf.constant(split.filepaths)
    labels_tensor = tf.constant(split.labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
    if shuffle:
        dataset = dataset.shuffle(len(split.filepaths), reshuffle_each_iteration=True)

    def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def build_datasets(config: Dict) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Builds train/validation/test datasets from configuration, returning
    `(train_ds, val_ds, test_ds, class_names)`.
    """
    data_cfg = config["data"]
    class_names: List[str] = data_cfg["species"]
    class_to_index = {name.lower(): idx for idx, name in enumerate(class_names)}

    base_dir = Path(config["project_root"])
    image_dir = base_dir / data_cfg["image_dir"]
    manifest_path = base_dir / data_cfg["labels_path"]
    split_manifest = base_dir / data_cfg["split_manifest_path"]

    records = _load_manifest(manifest_path, image_dir, class_to_index)

    if split_manifest.exists():
        saved = _load_split_manifest(split_manifest)
        required_keys = {"train", "val", "test"}
        if not required_keys.issubset(saved.keys()):
            raise ValueError(
                f"Split manifest at {split_manifest} is missing keys: {required_keys - set(saved.keys())}"
            )
        filename_to_label = {Path(path).name: label for path, label in records}
        for filename, label in filename_to_label.items():
            # Ensure manifest and split file are aligned.
            if label not in class_to_index.values():
                raise ValueError(f"Split manifest references unknown label index for {filename}.")
        train_files = [str(image_dir / name) for name in saved["train"]]
        val_files = [str(image_dir / name) for name in saved["val"]]
        test_files = [str(image_dir / name) for name in saved["test"]]
        train_labels = [filename_to_label[Path(path).name] for path in train_files]
        val_labels = [filename_to_label[Path(path).name] for path in val_files]
        test_labels = [filename_to_label[Path(path).name] for path in test_files]
        splits = (
            DatasetSplit(train_files, train_labels),
            DatasetSplit(val_files, val_labels),
            DatasetSplit(test_files, test_labels),
        )
    else:
        train_ratio = data_cfg.get("train_ratio", 0.7)
        val_ratio = data_cfg.get("val_ratio", 0.2)
        seed = data_cfg.get("seed", 42)
        splits = _stratified_split(records, train_ratio, val_ratio, seed)
        _save_split_manifest(
            split_manifest,
            {"train": splits[0], "val": splits[1], "test": splits[2]},
        )

    image_size = (data_cfg["image_size"], data_cfg["image_size"])
    batch_size = config["training"]["batch_size"]

    train_ds = _make_tf_dataset(splits[0], image_size, batch_size, shuffle=True)
    val_ds = _make_tf_dataset(splits[1], image_size, batch_size, shuffle=False)
    test_ds = _make_tf_dataset(splits[2], image_size, batch_size, shuffle=False)

    return train_ds, val_ds, test_ds, class_names
