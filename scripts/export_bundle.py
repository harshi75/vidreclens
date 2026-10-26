import argparse
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch

from day4_twotower import ALL, CONFIGS
from day7_clusters import TITLES, load_titles
from vidreclens.config import PROCESSED, REPORTS
from vidreclens.data.history import padded_history
from vidreclens.features.io import load_features
from vidreclens.models.twotower import TwoTower
from vidreclens.serve.engine import Recommender
from vidreclens.serve.export import export_bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--reports", type=Path, default=REPORTS)
    p.add_argument("--titles", type=Path, default=TITLES)
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/serving_bundle"))
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--config", default="hybrid_all4", choices=list(CONFIGS))
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--id-drop", dest="id_drop", type=float, default=0.3)
    p.add_argument("--mod-drop", dest="mod_drop", type=float, default=0.2)
    p.add_argument("--max-len", dest="max_len", type=int, default=50)
    p.add_argument("--checks", type=int, default=200)
    a = p.parse_args()
    ckpt = a.ckpt or a.reports / f"twotower_{a.config}_refit.pt"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found; run scripts/day5_bias.py first")

    df = pl.read_parquet(a.processed / "interactions.parquet")
    n_users = pl.read_parquet(a.processed / "users.parquet").height
    items_df = pl.read_parquet(a.processed / "items.parquet").sort("iid")
    n_items = items_df.height
    item_ids = items_df["item"].to_numpy()
    feats = {k: torch.as_tensor(v) for k, v in load_features(a.processed / "features", ALL).items()}
    names, use_id = CONFIGS[a.config]
    model = TwoTower({k: feats[k] for k in names}, n_items, a.dim, use_id, a.id_drop, a.mod_drop)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    seen = df.filter(pl.col("split").is_in(["train", "val"]))["iid"].to_numpy()
    pop = np.bincount(seen, minlength=n_items)
    titles = load_titles(a.titles, item_ids)
    out = export_bundle(model, item_ids, titles, pop, a.out, a.alpha, a.max_len)
    eng = Recommender(out)

    P = padded_history(df, ["train", "val"], n_users, n_items, a.max_len)
    with torch.no_grad():
        IR = model.encode_items()
    logpop = torch.log1p(torch.as_tensor(pop, dtype=torch.float32))
    same, worst = 0, 0.0
    n = min(a.checks, n_users)
    for u in range(n):
        hist = P[u][P[u] != n_items]
        h = torch.as_tensor(hist)[None]
        with torch.no_grad():
            uv = model.encode_users(IR, h, torch.ones_like(h, dtype=torch.bool))[0]
            s = eng.scale * (IR[:-1] @ uv) - a.alpha * logpop
            s[torch.as_tensor(hist)] = -float("inf")
            top = torch.topk(s, 10)
        ref = [int(item_ids[i]) for i in top.indices]
        res = eng.recommend([int(item_ids[i]) for i in hist], 10)
        got = [r["item_id"] for r in res["items"]]
        same += int(got == ref)
        worst = max(worst, float(np.abs(np.array([r["score"] for r in res["items"]]) - top.values.numpy()).max()))
    print(f"parity check on {n} users: identical top-10 for {same}/{n}, largest score difference {worst:.2e}")
    zp = shutil.make_archive(str(out), "zip", out)
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"bundle {out} ({size:.1f} MB uncompressed) -> {zp}")


if __name__ == "__main__":
    main()
