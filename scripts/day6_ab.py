import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from day4_twotower import ALL, fit, score_fn, split_ctx, CONFIGS
from vidreclens.config import PROCESSED, REPORTS
from vidreclens.eval.metrics import tie_aware_rank
from vidreclens.features.io import load_features
from vidreclens.models.baselines import itemknn
from vidreclens.models.twotower import TwoTower

ZA, ZB = 1.959964, 0.841621


def user_ranks(fn, target, n_users, batch):
    ranks = np.empty(n_users)
    for a0 in range(0, n_users, batch):
        idx = np.arange(a0, min(a0 + batch, n_users))
        ranks[idx] = tie_aware_rank(fn(idx), target[idx])
    return ranks


def outcomes(r, k=10):
    hit = r <= k
    return hit.astype(np.float64), np.where(hit, 1 / np.log2(r + 1), 0.0)


def pvalue(z):
    return math.erfc(abs(z) / math.sqrt(2))


def mean_ci(x):
    m = float(x.mean())
    h = 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return m, m - h, m + h


def paired(a, b):
    d = b - a
    m = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(len(d))
    z = m / se if se > 0 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se, pvalue(z)


def required_n(p1, p2):
    return (ZA + ZB) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2


def simulate(pairs, n_arm, sims, rng):
    n = len(next(iter(pairs.values()))[0])
    sig = {k: np.zeros(sims, dtype=bool) for k in pairs}
    lift = {k: np.zeros(sims) for k in pairs}
    for s in range(sims):
        idx = rng.permutation(n)[: 2 * n_arm]
        c, t = idx[:n_arm], idx[n_arm:]
        for k, (ha, hb) in pairs.items():
            pc, pt = ha[c].mean(), hb[t].mean()
            se = math.sqrt((pc * (1 - pc) + pt * (1 - pt)) / n_arm)
            sig[k][s] = se > 0 and abs(pt - pc) / se > ZA
            lift[k][s] = (pt - pc) / pc if pc > 0 else 0.0
    return {
        k: {
            "power": float(sig[k].mean()),
            "lift_p5": float(np.percentile(lift[k], 5)),
            "lift_median": float(np.median(lift[k])),
            "lift_p95": float(np.percentile(lift[k], 95)),
        }
        for k in pairs
    }


def segments(labels, names, hits, a_key, b_key):
    rows = []
    for i, nm in enumerate(names):
        m = labels == i
        if not m.any():
            continue
        d = paired(hits[a_key][m], hits[b_key][m]) if m.sum() > 1 else (0.0, 0.0, 0.0, 1.0)
        rows.append({"segment": nm, "users": int(m.sum()), **{k: float(v[m].mean()) for k, v in hits.items()}, "diff": d[0], "p": d[3]})
    return rows


def print_segments(title, rows, keys, a_key, b_key):
    print(f"\n{title}")
    print(f"{'segment':<10}{'users':>9}" + "".join(f"{k:>12}" for k in keys) + f"{b_key + '-' + a_key:>20}{'p':>10}")
    for r in rows:
        print(f"{r['segment']:<10}{r['users']:>9}" + "".join(f"{r[k]:>12.4f}" for k in keys) + f"{r['diff']:>+20.4f}{r['p']:>10.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--reports", type=Path, default=REPORTS)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--config", default="hybrid_all4", choices=list(CONFIGS))
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--epochs", type=int, default=22)
    p.add_argument("--sims", type=int, default=1000)
    p.add_argument("--arm-sizes", dest="arm_sizes", type=int, nargs="+", default=[5000, 10000, 20000, 35000])
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
    a.reports.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    ckpt = a.ckpt or a.reports / f"twotower_{a.config}_refit.pt"

    df = pl.read_parquet(a.processed / "interactions.parquet")
    n_users = pl.read_parquet(a.processed / "users.parquet").height
    n_items = pl.read_parquet(a.processed / "items.parquet").height
    feats = {k: torch.as_tensor(v, device=dev) for k, v in load_features(a.processed / "features", ALL).items()}
    test = split_ctx(df, "test", ["train", "val"], n_users, n_items, a.max_len, dev)

    names, use_id = CONFIGS[a.config]
    if ckpt.exists():
        model = TwoTower({k: feats[k] for k in names}, n_items, a.dim, use_id, a.id_drop, a.mod_drop).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f"loaded {ckpt}")
    else:
        print(f"{ckpt} not found, training {a.config} on train+val for {a.epochs} epochs")
        model = fit(a.config, CONFIGS[a.config], feats, n_items, test, a, dev, epochs=a.epochs)

    base = score_fn(model, test, n_items)
    scale = min(model.log_scale.exp().item(), 100.0)
    logpop = np.log1p(np.asarray(test["H"].sum(0)).ravel()).astype(np.float32)
    tag = f"tt_a{a.alpha:g}"
    policies = {
        "itemknn": itemknn(test["H"]),
        "tt_a0": lambda idx: base(idx) * scale,
        tag: lambda idx: base(idx) * scale - a.alpha * logpop,
    }
    ranks = {}
    for k, fn in policies.items():
        ranks[k] = user_ranks(fn, test["target"], n_users, a.eval_batch)
        print(f"scored {k}", flush=True)
    hit, ndcg = {}, {}
    for k, r in ranks.items():
        hit[k], ndcg[k] = outcomes(r)

    res = {"n_users": n_users, "alpha": a.alpha}
    print(f"\n== 1. Policy metrics on {n_users} test users (95% CI) ==")
    res["metrics"] = {}
    for k in policies:
        h, n_ = mean_ci(hit[k]), mean_ci(ndcg[k])
        res["metrics"][k] = {"hit@10": h, "ndcg@10": n_}
        print(f"{k:<10} hit@10 {h[0]:.4f} [{h[1]:.4f}, {h[2]:.4f}]   ndcg@10 {n_[0]:.4f} [{n_[1]:.4f}, {n_[2]:.4f}]")

    comps = [("tt_a0", tag), ("itemknn", tag), ("itemknn", "tt_a0")]
    print("\n== 2. Paired comparison, same users under both policies (B minus A) ==")
    res["paired"] = {}
    for ka, kb in comps:
        dh, dn = paired(hit[ka], hit[kb]), paired(ndcg[ka], ndcg[kb])
        only_b = int(((hit[kb] == 1) & (hit[ka] == 0)).sum())
        only_a = int(((hit[kb] == 0) & (hit[ka] == 1)).sum())
        res["paired"][f"{ka}->{kb}"] = {"hit": dh, "ndcg": dn, "only_b": only_b, "only_a": only_a}
        print(f"A={ka:<8} B={kb:<8} hit@10 {dh[0]:+.4f} [{dh[1]:+.4f}, {dh[2]:+.4f}] p={dh[3]:.4f}   ndcg@10 {dn[0]:+.4f} [{dn[1]:+.4f}, {dn[2]:+.4f}] p={dn[3]:.4f}   hits only A/only B: {only_a}/{only_b}")

    sizes = sorted({s for s in a.arm_sizes if s <= n_users // 2} | {n_users // 2})
    pairs = {f"{ka}->{kb}": (hit[ka], hit[kb]) for ka, kb in comps}
    rng = np.random.default_rng(a.seed)
    print(f"\n== 3. Simulated A/B test: users split at random, each sees only its arm's policy ({a.sims} simulations per size) ==")
    print(f"{'users/arm':>10}  {'comparison':<22}{'power':>8}{'lift p5':>10}{'lift med':>10}{'lift p95':>10}")
    res["ab"] = {}
    for s in sizes:
        out = simulate(pairs, s, a.sims, rng)
        res["ab"][s] = out
        for k, v in out.items():
            print(f"{s:>10}  {k:<22}{v['power']:>8.2f}{v['lift_p5']:>+10.1%}{v['lift_median']:>+10.1%}{v['lift_p95']:>+10.1%}")

    p1, p2 = float(hit["tt_a0"].mean()), float(hit[tag].mean())
    print(f"\n== 4. Users needed per arm (80% power, 5% two-sided) when the baseline hit rate is {p1:.4f} ==")
    res["required"] = {}
    for rel in (0.02, 0.05, 0.10, 0.20):
        n_req = required_n(p1, p1 * (1 + rel))
        res["required"][rel] = n_req
        print(f"relative lift {rel:>4.0%}: {n_req:>10,.0f}")
    obs = required_n(p1, p2)
    res["required_observed"] = obs
    print(f"observed lift {p2 / p1 - 1:+.1%} ({p1:.4f} -> {p2:.4f}): {obs:,.0f} per arm; available {n_users // 2:,} per arm")

    hist_n = np.asarray(test["H"].sum(1)).ravel()
    hl = np.select([hist_n <= 4, hist_n <= 6, hist_n <= 10], [0, 1, 2], 3)
    tpop = np.asarray(test["H"].sum(0)).ravel()[test["target"]]
    tl = np.select([tpop == 0, tpop <= 4, tpop <= 19, tpop <= 99], [0, 1, 2, 3], 4)
    keys = ["itemknn", "tt_a0", tag]
    res["segments"] = {
        "history_length": segments(hl, ["4", "5-6", "7-10", "11+"], {k: hit[k] for k in keys}, "itemknn", tag),
        "target_popularity": segments(tl, ["0", "1-4", "5-19", "20-99", "100+"], {k: hit[k] for k in keys}, "itemknn", tag),
    }
    print_segments("== 5a. hit@10 by length of the user's history ==", res["segments"]["history_length"], keys, "itemknn", tag)
    print_segments("== 5b. hit@10 by how many past interactions the target video had (0 = cold item) ==", res["segments"]["target_popularity"], keys, "itemknn", tag)

    (a.reports / "day6_ab.json").write_text(json.dumps(res, indent=2, default=float))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in pairs:
        ax.plot(sizes, [res["ab"][s][k]["power"] for s in sizes], marker="o", label=k)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("users per arm")
    ax.set_ylabel("share of simulations with p < 0.05")
    ax.set_title("Simulated A/B test power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.reports / "ab_power.png", dpi=150)
    print("\nsaved day6_ab.json and ab_power.png")


if __name__ == "__main__":
    main()
