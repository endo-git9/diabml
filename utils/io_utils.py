# utils/io_utils.py
import os
import joblib
import json
import numpy as np

def ensure_dir(path):

    os.makedirs(path, exist_ok=True)

def save_json(obj, path):

    def convert(o):
        # primitive numpy -> python
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return list(o)
        # fallback: string representation (aman)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert, ensure_ascii=False)

def load_model(path):
    """Muat model dari file .pkl"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)
