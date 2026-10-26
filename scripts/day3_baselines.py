import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from vidreclens.config import PROCESSED, REPORTS
from vidreclens.eval.metrics import evaluate
from vidreclens.features.io import load_features
from vidreclens.models.baselines import content_profile, history_matrix, itemknn, popularity

NAMES = ["clip_vitb32_covers", "videomae", "cliprn50", "bgem3"]
COLS = ["recall@10", "ndcg@10", "recall@50", "ndcg@50", "recall@10_head", "recall@10_tail"]


def show(rows):
    print(f"{'model':<26}" + "".join(f"{c:>16}" for c in COLS))
    for name, r in rows.items():
        print(f"{name:<26}" + "".join(f"{r[c]:>16.4f}" for c in COLS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--reports", type=Path, default=REPORTS)
    p.add_argument("--batch", type=int, default=2048)
    a = p.parse_args()
    a.reports.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(a.processed / "interactions.parquet")
    n_users = pl.read_parquet(a.processed / "users.parquet").height
    n_items = pl.read_parquet(a.processed / "items.parquet").height
    feats = load_features(a.processed / "features", NAMES)
    feats["content_all"] = np.concatenate([feats[n] for n in NAMES], axis=1) / np.sqrt(len(NAMES))

    for split, hist in [("val", ["train"]), ("test", ["train", "val"])]:
        H = history_matrix(df, n_users, n_items, hist)
        t = df.filter(pl.col("split") == split).sort("uid")
        target = np.zeros(n_users, dtype=np.int64)
        target[t["uid"].to_numpy()] = t["iid"].to_numpy()
        pop = np.asarray(H.sum(0)).ravel()
        head = np.zeros(n_items, dtype=bool)
        head[np.argsort(-pop)[: int(0.2 * n_items)]] = True

        models = {"popularity": popularity(H), "itemknn": itemknn(H)}
        for n, E in feats.items():
            models[f"content:{n}"] = content_profile(H, E)

        rows = {}
        for name, fn in models.items():
            rows[name] = evaluate(fn, target, head, a.batch)
            print(split, name, round(rows[name]["recall@10"], 4), flush=True)
        print(f"\n== {split} (history: {'+'.join(hist)}, random recall@10 = {10 / n_items:.5f}) ==")
        show(rows)
        (a.reports / f"baselines_{split}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
