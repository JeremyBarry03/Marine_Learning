"""Utility helpers for managing benthic image assets."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


def consolidate_species_images(
    source_dirs: Iterable[Path],
    destination_dir: Path,
    species_name: str,
) -> None:
    """
    Copies images from multiple acquisition runs into a single destination folder.

    Parameters
    ----------
    source_dirs:
        Iterable of directories containing raw captures for the same species.
    destination_dir:
        Target directory where consolidated files will be placed.
    species_name:
        Name of the species; used to generate deterministic filenames.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    for source_dir in source_dirs:
        for image_path in sorted(source_dir.glob("*")):
            if not image_path.is_file():
                continue
            counter += 1
            new_name = f"{counter:06d}_{species_name.lower()}{image_path.suffix}"
            shutil.copy2(image_path, destination_dir / new_name)
    print(f"Consolidated {counter} images into {destination_dir}")


if __name__ == "__main__":
    print(
        "This module provides helper functions for dataset maintenance. "
        "Invoke consolidate_species_images from another script or a REPL as needed."
    )

