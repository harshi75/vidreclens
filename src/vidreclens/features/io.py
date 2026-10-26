from pathlib import Path

import numpy as np


def load_features(feat_dir: Path, names) -> dict:
    out = {}
    for n in names:
        x = np.load(Path(feat_dir) / f"{n}.npy").astype(np.float32)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        out[n] = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    return out
