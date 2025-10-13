"""Quick manual test harness for the Flask inference API."""
from pathlib import Path

import requests

from src.config import MARINE_CONFIG

BACKEND_URL = "http://127.0.0.1:5000"

root = Path(MARINE_CONFIG["project_root"])
labels_path = root / MARINE_CONFIG["data"]["labels_path"]
image_dir = root / MARINE_CONFIG["data"]["image_dir"]

with labels_path.open("r", encoding="utf-8") as handle:
    first_line = handle.readline().strip()
filename = first_line.split()[0]
image_path = image_dir / filename

with image_path.open("rb") as image_file:
    files = {"file": image_file}
    response = requests.post(f"{BACKEND_URL}/predict", files=files)

print(response.status_code)
print(response.json())
