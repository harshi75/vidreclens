import argparse
from pathlib import Path

import numpy as np
import polars as pl

from vidreclens.config import PROCESSED

NAMES = ["clip_vitb32_covers", "videomae", "cliprn50", "bgem3"]


def aligned(emb_dir: Path, name: str, items: np.ndarray) -> np.ndarray:
    v = np.load(emb_dir / f"{name}.npy").astype(np.float32)
    ids = np.load(emb_dir / f"{name}_ids.npy")
    order = np.argsort(ids)
    srt = ids[order]
    idx = np.clip(np.searchsorted(srt, items), 0, len(srt) - 1)
    missing = int((srt[idx] != items).sum())
    if missing:
        raise ValueError(f"{name}: {missing} items without features")
    return v[order[idx]]


def topk(x: np.ndarray, sample: np.ndarray, k: int) -> np.ndarray:
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    s = x[sample] @ x.T
    s[np.arange(len(sample)), sample] = -np.inf
    return np.argpartition(-s, k, axis=1)[:, :k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", type=Path, default=PROCESSED.parent / "embeddings")
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()

    items = pl.read_parquet(a.processed / "items.parquet").sort("iid")["item"].to_numpy()
    out = a.processed / "features"
    out.mkdir(exist_ok=True)

    feats = {}
    for n in NAMES:
        feats[n] = aligned(a.emb, n, items)
        np.save(out / f"{n}.npy", feats[n].astype(np.float16))
        print(n, feats[n].shape)

    rng = np.random.default_rng(0)
    sample = rng.choice(len(items), size=min(500, len(items)), replace=False)
    nn = {n: topk(f, sample, a.k) for n, f in feats.items()}
    print("neighbour overlap@%d between modalities" % a.k)
    for i, m in enumerate(NAMES):
        for n in NAMES[i + 1 :]:
            ov = np.mean([len(set(x) & set(y)) / a.k for x, y in zip(nn[m], nn[n])])
            print(f"  {m} vs {n}: {ov:.3f}")


if __name__ == "__main__":
    main()
