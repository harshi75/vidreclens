import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from day4_twotower import ALL, CONFIGS, fit, score_fn, split_ctx
from vidreclens.config import PROCESSED, REPORTS
from vidreclens.data.eda import gini
from vidreclens.eval.metrics import summarize, tie_aware_rank
from vidreclens.features.io import load_features
from vidreclens.models.baselines import itemknn

COLS = ["recall@10", "ndcg@10", "recall@50", "recall@10_head", "recall@10_tail", "coverage@10", "head_share@10", "gini@10"]


def run(fn, ctx, n_users, n_items, batch, k=10):
    target = ctx["target"][:n_users]
    ranks = np.empty(n_users)
    counts = np.zeros(n_items)
    for a0 in range(0, n_users, batch):
        idx = np.arange(a0, min(a0 + batch, n_users))
        s = fn(idx)
        ranks[idx] = tie_aware_rank(s, target[idx])
        top = np.argpartition(-s, k, axis=1)[:, :k]
        counts += np.bincount(top.ravel(), minlength=n_items)
    out = summarize(ranks, ctx["head"][target])
    out["coverage@10"] = float((counts > 0).mean())
    out["head_share@10"] = float(counts[ctx["head"]].sum() / counts.sum())
    out["gini@10"] = gini(counts)
    return out


def sweep(model, ctx, alphas, n_users, n_items, batch):
    base = score_fn(model, ctx, n_items)
    scale = min(model.log_scale.exp().item(), 100.0)
    logpop = np.log1p(np.asarray(ctx["H"].sum(0)).ravel()).astype(np.float32)
    rows = {}
    for al in alphas:
        rows[al] = run(lambda idx, al=al: base(idx) * scale - al * logpop, ctx, n_users, n_items, batch)
        print(f"alpha {al}: recall@10 {rows[al]['recall@10']:.4f} tail {rows[al]['recall@10_tail']:.4f} coverage {rows[al]['coverage@10']:.3f}", flush=True)
    return rows


def table(rows):
    print(f"{'model':<14}" + "".join(f"{c:>16}" for c in COLS))
    for name, r in rows.items():
        print(f"{name:<14}" + "".join(f"{r[c]:>16.4f}" for c in COLS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--reports", type=Path, default=REPORTS)
    p.add_argument("--config", default="hybrid_all4", choices=list(CONFIGS))
    p.add_argument("--epochs", type=int, default=22)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--tol", type=float, default=0.03)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--id-drop", dest="id_drop", type=float, default=0.3)
    p.add_argument("--mod-drop", dest="mod_drop", type=float, default=0.2)
    p.add_argument("--max-len", dest="max_len", type=int, default=50)
    p.add_argument("--val-users", dest="val_users", type=int, default=30000)
    p.add_argument("--eval-batch", dest="eval_batch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    assert a.alphas[0] == 0.0
    a.reports.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    df = pl.read_parquet(a.processed / "interactions.parquet")
    n_users = pl.read_parquet(a.processed / "users.parquet").height
    n_items = pl.read_parquet(a.processed / "items.parquet").height
    feats = {k: torch.as_tensor(v, device=dev) for k, v in load_features(a.processed / "features", ALL).items()}
    val = split_ctx(df, "val", ["train"], n_users, n_items, a.max_len, dev)
    test = split_ctx(df, "test", ["train", "val"], n_users, n_items, a.max_len, dev)

    m1 = fit(a.config, CONFIGS[a.config], feats, n_items, val, a, dev, epochs=a.epochs)
    torch.save(m1.state_dict(), a.reports / f"twotower_{a.config}_trainonly.pt")
    kval = min(a.val_users, n_users)
    val_rows = sweep(m1, val, a.alphas, kval, n_items, a.eval_batch)
    ok = [al for al in a.alphas if val_rows[al]["recall@10"] >= (1 - a.tol) * val_rows[0.0]["recall@10"]]
    chosen = max(ok)

    m2 = fit(a.config, CONFIGS[a.config], feats, n_items, test, a, dev, epochs=a.epochs)
    torch.save(m2.state_dict(), a.reports / f"twotower_{a.config}_refit.pt")
    test_rows = sweep(m2, test, a.alphas, n_users, n_items, a.eval_batch)
    knn = run(itemknn(test["H"]), test, n_users, n_items, a.eval_batch)

    print(f"\n== val (train-only model, first {kval} users) ==")
    table({f"a={al}": r for al, r in val_rows.items()})
    print("\n== test (refit on train+val, all users) ==")
    table({"itemknn": knn} | {f"a={al}": r for al, r in test_rows.items()})
    print(f"\nchosen alpha = {chosen} (largest alpha with val recall@10 within {a.tol:.0%} of alpha=0)")
    r0, rc = test_rows[0.0], test_rows[chosen]
    print(f"test at chosen alpha vs alpha=0: recall@10 {r0['recall@10']:.4f} -> {rc['recall@10']:.4f}, tail recall@10 {r0['recall@10_tail']:.4f} -> {rc['recall@10_tail']:.4f}, coverage@10 {r0['coverage@10']:.3f} -> {rc['coverage@10']:.3f}")
    (a.reports / "bias_val.json").write_text(json.dumps({str(k): v for k, v in val_rows.items()}, indent=2))
    (a.reports / "bias_test.json").write_text(json.dumps({"itemknn": knn} | {str(k): v for k, v in test_rows.items()}, indent=2))


if __name__ == "__main__":
    main()
