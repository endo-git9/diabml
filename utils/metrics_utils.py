import os
import joblib
import json
from datetime import datetime


def ensure_dir(path: str):
    """Ensure a directory exists."""
    if not path:
        raise ValueError("Directory path cannot be empty.")
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str, backup: bool = True):
    """Save a JSON file with optional backup."""
    ensure_dir(os.path.dirname(path))
    if backup and os.path.exists(path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.rename(path, f"{path}.bak_{timestamp}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {path}")


def load_model(path: str):
    """Load ML model safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"[LOADED] {path}")
    return joblib.load(path)
