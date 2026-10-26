import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from vidreclens.config import PROCESSED, REPORTS
from vidreclens.data.history import padded_history
from vidreclens.eval.metrics import evaluate
from vidreclens.features.io import load_features
from vidreclens.models.baselines import history_matrix, mask_history
from vidreclens.models.twotower import TwoTower

ALL = ["clip_vitb32_covers", "videomae", "cliprn50", "bgem3"]
NO_VM = [n for n in ALL if n != "videomae"]
CONFIGS = {
    "id_only": ([], True),
    "content_only": (NO_VM, False),
    "hybrid_3mod": (NO_VM, True),
    "hybrid_all4": (ALL, True),
}
COLS = ["recall@10", "ndcg@10", "recall@50", "ndcg@50", "recall@10_head", "recall@10_tail"]


def table(rows):
    print(f"{'model':<26}" + "".join(f"{c:>16}" for c in COLS))
    for name, r in rows.items():
        print(f"{name:<26}" + "".join(f"{r[c]:>16.4f}" for c in COLS))


def split_ctx(df, split, hist, n_users, n_items, max_len, dev):
    H = history_matrix(df, n_users, n_items, hist)
    t = df.filter(pl.col("split") == split).sort("uid")
    target = np.zeros(n_users, dtype=np.int64)
    target[t["uid"].to_numpy()] = t["iid"].to_numpy()
    pop = np.asarray(H.sum(0)).ravel()
    head = np.zeros(n_items, dtype=bool)
    head[np.argsort(-pop)[: int(0.2 * n_items)]] = True
    P = torch.as_tensor(padded_history(df, hist, n_users, n_items, max_len), device=dev)
    return {"H": H, "target": target, "head": head, "P": P}


def score_fn(model, ctx, pad):
    model.eval()
    with torch.no_grad():
        IR = model.encode_items()

    def fn(idx):
        hist = ctx["P"][torch.as_tensor(idx, device=IR.device)]
        with torch.no_grad():
            s = (model.encode_users(IR, hist, hist != pad) @ IR[:-1].T).float().cpu().numpy()
        return mask_history(s, ctx["H"], idx)

    return fn


def fit(name, cfg, feats, n_items, tr, a, dev, epochs=None):
    names, use_id = cfg
    select = epochs is None
    n_ep = epochs or a.epochs
    torch.manual_seed(a.seed)
    model = TwoTower({k: feats[k] for k in names}, n_items, a.dim, use_id, a.id_drop, a.mod_drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    pad = n_items
    P = tr["P"]
    ru, rp = (torch.as_tensor(x, device=dev) for x in np.nonzero(P.cpu().numpy() != pad))
    ar = torch.arange(P.size(1), device=dev)
    kval = min(a.val_users, len(tr["target"]))
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, n_ep + 1):
        model.train()
        perm = torch.randperm(len(ru), device=dev)
        total = torch.zeros((), device=dev)
        for b in range(0, len(perm), a.batch):
            sel = perm[b : b + a.batch]
            u, p = ru[sel], rp[sel]
            hist = P[u]
            target = hist.gather(1, p[:, None]).squeeze(1)
            keep = (hist != pad) & (ar[None, :] != p[:, None])
            IR = model.encode_items()
            uv = model.encode_users(IR, hist, keep)
            logits = model.log_scale.exp().clamp(max=100.0) * (uv @ IR.T)
            logits = logits.scatter(1, torch.where(keep, hist, torch.full_like(hist, pad)), float("-inf"))
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * len(sel)
        if not select:
            if ep % 5 == 0 or ep == n_ep:
                print(f"{name} refit ep{ep} loss {total.item() / len(ru):.4f} ({time.time() - t0:.0f}s)", flush=True)
            continue
        if ep % a.eval_every == 0 or ep == n_ep:
            r = evaluate(score_fn(model, tr, pad), tr["target"][:kval], tr["head"], a.eval_batch)["recall@10"]
            print(f"{name} ep{ep} loss {total.item() / len(ru):.4f} val_r@10 {r:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if r > best:
                best, best_state, best_ep, bad = r, copy.deepcopy(model.state_dict()), ep, 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
    if select:
        model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--reports", type=Path, default=REPORTS)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--id-drop", dest="id_drop", type=float, default=0.3)
    p.add_argument("--mod-drop", dest="mod_drop", type=float, default=0.2)
    p.add_argument("--max-len", dest="max_len", type=int, default=50)
    p.add_argument("--eval-every", dest="eval_every", type=int, default=2)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val-users", dest="val_users", type=int, default=10000)
    p.add_argument("--eval-batch", dest="eval_batch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refit", nargs="*", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    a.reports.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    df = pl.read_parquet(a.processed / "interactions.parquet")
    n_users = pl.read_parquet(a.processed / "users.parquet").height
    n_items = pl.read_parquet(a.processed / "items.parquet").height
    feats = {k: torch.as_tensor(v, device=dev) for k, v in load_features(a.processed / "features", ALL).items()}
    val = split_ctx(df, "val", ["train"], n_users, n_items, a.max_len, dev)
    test = split_ctx(df, "test", ["train", "val"], n_users, n_items, a.max_len, dev)

    if a.refit is not None:
        res = {}
        for kv in a.refit:
            name, ep = kv.split("=")
            model = fit(name, CONFIGS[name], feats, n_items, test, a, dev, epochs=int(ep))
            res[f"refit:{name}"] = evaluate(score_fn(model, test, n_items), test["target"], test["head"], a.eval_batch)
            print(name, "refit test r@10", round(res[f"refit:{name}"]["recall@10"], 4), flush=True)
        rows = {}
        for fname in ("baselines_test.json", "twotower_test.json"):
            f = a.reports / fname
            if f.exists():
                prev = json.loads(f.read_text())
                keep = ("popularity", "itemknn", "content:content_all") if fname.startswith("baselines") else tuple(prev)
                rows |= {k: prev[k] for k in keep if k in prev}
        rows |= res
        print("\n== test (refit on train+val) ==")
        table(rows)
        (a.reports / "twotower_test_refit.json").write_text(json.dumps(res, indent=2))
        return

    out = {"val": {}, "test": {}}
    for name in a.configs:
        model = fit(name, CONFIGS[name], feats, n_items, val, a, dev)
        for split, ctx in (("val", val), ("test", test)):
            out[split][f"twotower:{name}"] = evaluate(score_fn(model, ctx, n_items), ctx["target"], ctx["head"], a.eval_batch)
        torch.save(model.state_dict(), a.reports / f"twotower_{name}.pt")
        print(name, "test r@10", round(out["test"][f"twotower:{name}"]["recall@10"], 4), flush=True)

    for split in ("val", "test"):
        rows = dict(out[split])
        f = a.reports / f"baselines_{split}.json"
        if f.exists():
            base = json.loads(f.read_text())
            rows = {k: base[k] for k in ("popularity", "itemknn", "content:content_all") if k in base} | rows
        print(f"\n== {split} ==")
        table(rows)
        (a.reports / f"twotower_{split}.json").write_text(json.dumps(out[split], indent=2))


if __name__ == "__main__":
    main()
