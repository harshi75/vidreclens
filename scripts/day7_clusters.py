import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import torch

from day4_twotower import ALL, CONFIGS, fit, score_fn, split_ctx
from vidreclens.config import PROCESSED, REPORTS
from vidreclens.eval.metrics import tie_aware_rank
from vidreclens.features.io import load_features
from vidreclens.models.baselines import itemknn
from vidreclens.models.twotower import TwoTower

CONTENT = [n for n in ALL if n != "videomae"]
TITLES = Path("/kaggle/working/data/MicroLens-100k_title_en.csv")


def load_titles(path, items):
    titles = {}
    if path.exists():
        for line in path.open(encoding="utf-8", errors="ignore"):
            m = re.match(r"\s*(\d+)\s*,\s*(.*)", line)
            if m:
                titles[int(m.group(1))] = m.group(2).strip()
    return [titles.get(int(i), "") for i in items]


def cluster_items(feats, k, seed):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = np.concatenate([feats[n] for n in CONTENT], axis=1)
    Z = PCA(n_components=64, random_state=seed).fit_transform(X)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-8)
    return KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Z).labels_, Z


def score_policy(fn, target, n_users, n_items, batch, k=10):
    ranks = np.empty(n_users)
    counts = np.zeros(n_items)
    for a0 in range(0, n_users, batch):
        idx = np.arange(a0, min(a0 + batch, n_users))
        s = fn(idx)
        ranks[idx] = tie_aware_rank(s, target[idx])
        counts += np.bincount(np.argpartition(-s, k, axis=1)[:, :k].ravel(), minlength=n_items)
    return ranks, counts


def short(t, n=60):
    return t if len(t) <= n else t[: n - 1] + "…"


def distinctive_words(titles, labels, K, top=8, min_count=15):
    from collections import Counter

    overall, per = Counter(), [Counter() for _ in range(K)]
    for t, c in zip(titles, labels):
        w = set(re.findall(r"[a-z]{3,}", t.lower()))
        overall.update(w)
        per[c].update(w)
    n, sizes = len(titles), np.bincount(labels, minlength=K)
    out = []
    for c in range(K):
        sc = {w: (k / sizes[c]) / (overall[w] / n) for w, k in per[c].items() if overall[w] >= min_count and k >= 5}
        out.append([w for w, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:top]])
    return out


def name_prompt(rows, words):
    lines = [
        "Below, for each cluster of short videos, are example titles (machine-translated from Chinese, so the wording is rough) and words that are unusually frequent in the cluster's titles.",
        "For each cluster write exactly one line: 'C<number>: <2 to 4 word topic>'. The topic must be justified by the titles or words. If they share no clear topic, write 'C<number>: mixed'. Output only these lines.",
        "",
    ]
    for r, w in zip(rows, words):
        lines.append(f"C{r['cluster']} | words: {', '.join(w) or 'none'} | titles: " + "; ".join(short(t, 70) for t in r["titles"][:5]))
    return "\n".join(lines)


def rewrite_prompt(facts):
    return (
        "Rewrite the facts below as one short paragraph of at most 5 sentences for a product manager. "
        "Rules: use only the facts below; every number you write must appear in the facts; refer to clusters as C0, C1 and so on, never as categories; write each change as 'from X to Y'; do not use evaluative words such as significantly or notably; do not explain causes; do not mention genres, users' feelings or anything not stated.\n\nFacts:\n"
        + "\n".join("- " + f for f in facts)
    )


def unsupported_numbers(text, facts):
    ok = set(re.findall(r"\d+(?:\.\d+)?", facts)) | {"10"}
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return sorted(set(re.findall(r"\d+(?:\.\d+)?", text)) - ok)


def clean_names(text):
    lines = [re.sub(r"^(C\d+):\s*C?#\s*", r"\1: ", ln.strip()) for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def run_llm(name, prompts, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32).to(dev).eval()
    outs = []
    for pr in prompts:
        msgs = [
            {"role": "system", "content": "You are a careful analyst. You only use facts given to you and never invent numbers or causes."},
            {"role": "user", "content": pr},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=400, do_sample=False)
        outs.append(tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip())
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--reports", type=Path, default=REPORTS)
    p.add_argument("--titles", type=Path, default=TITLES)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--config", default="hybrid_all4", choices=list(CONFIGS))
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--epochs", type=int, default=22)
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
    items_df = pl.read_parquet(a.processed / "items.parquet").sort("iid")
    n_items = items_df.height
    stats = pl.read_parquet(a.processed / "item_stats.parquet").sort("iid")
    feats_np = load_features(a.processed / "features", ALL)
    feats = {k: torch.as_tensor(v, device=dev) for k, v in feats_np.items()}
    test = split_ctx(df, "test", ["train", "val"], n_users, n_items, a.max_len, dev)
    titles = load_titles(a.titles, items_df["item"].to_numpy())

    labels, Z = cluster_items(feats_np, a.k, a.seed)
    K = a.k
    words = distinctive_words(titles, labels, K)

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
    policies = {
        "itemknn": itemknn(test["H"]),
        "tt_a0": lambda idx: base(idx) * scale,
        "adj": lambda idx: base(idx) * scale - a.alpha * logpop,
    }
    target = test["target"]
    hit, counts = {}, {}
    for k, fn in policies.items():
        r, c = score_policy(fn, target, n_users, n_items, a.eval_batch)
        hit[k], counts[k] = (r <= 10).astype(float), c
        print(f"scored {k}", flush=True)

    tl = labels[target]
    demand = np.bincount(tl, minlength=K) / n_users
    expo = {k: np.bincount(labels, weights=c, minlength=K) / c.sum() for k, c in counts.items()}
    tvd = {k: float(0.5 * np.abs(expo[k] - demand).sum()) for k in policies}
    catalog = np.bincount(labels, minlength=K) / n_items
    tvdc = {k: float(0.5 * np.abs(expo[k] - catalog).sum()) for k in policies}
    hitc = {k: np.array([hit[k][tl == c].mean() if (tl == c).any() else np.nan for c in range(K)]) for k in policies}
    users_c = np.bincount(tl, minlength=K)
    log_views = stats["log_views"].to_numpy()
    like_rate = stats["like_rate"].to_numpy()
    tail = ~test["head"]

    rows = []
    for c in range(K):
        idx = np.where(labels == c)[0]
        cen = Z[idx].mean(0)
        rep = idx[np.argsort(-(Z[idx] @ cen))[:5]]
        rows.append(
            {
                "cluster": c,
                "items": int(len(idx)),
                "catalog_pct": 100 * len(idx) / n_items,
                "demand_pct": 100 * float(demand[c]),
                "tail_item_pct": 100 * float(tail[idx].mean()),
                "mean_log_views": float(log_views[idx].mean()),
                "mean_like_rate": float(like_rate[idx].mean()),
                "users": int(users_c[c]),
                "expo_pct": {k: 100 * float(expo[k][c]) for k in policies},
                "ratio": {k: float(expo[k][c] / max(demand[c], 1e-9)) for k in policies},
                "hit": {k: float(hitc[k][c]) for k in policies},
                "titles": [titles[i] or f"(item {items_df['item'][int(i)]})" for i in rep],
                "words": words[c],
            }
        )

    print(f"\n== Clusters: composition and demand (k={K}, KMeans on PCA of cover + title + CLIP RN50 embeddings) ==")
    print(f"{'cluster':<8}{'items':>7}{'catalog%':>10}{'demand%':>9}{'tail items%':>13}{'log views':>11}{'like rate':>11}   example titles")
    for r in rows:
        print(f"C{r['cluster']:<7}{r['items']:>7}{r['catalog_pct']:>10.1f}{r['demand_pct']:>9.1f}{r['tail_item_pct']:>13.1f}{r['mean_log_views']:>11.2f}{r['mean_like_rate']:>11.3f}   " + " | ".join(short(t, 28) for t in r["titles"][:2]))
    print("\n== Distinctive title words by cluster (much more frequent in this cluster than overall) ==")
    for c, w in enumerate(words):
        print(f"C{c:<4}" + ", ".join(w))
    print(f"\n== Policy KPIs by cluster (exposure = share of all top-10 slots; ratio = exposure / demand; hit@10 for users whose next video is in the cluster) ==")
    print(f"{'cluster':<8}{'users':>7}  {'exposure% knn / tt / adj':<28}{'ratio tt / adj':<18}{'hit@10 knn / tt / adj'}")
    for r in rows:
        e, q, h = r["expo_pct"], r["ratio"], r["hit"]
        print(f"C{r['cluster']:<7}{r['users']:>7}  {e['itemknn']:>6.1f} {e['tt_a0']:>6.1f} {e['adj']:>6.1f}          {q['tt_a0']:>5.2f} {q['adj']:>5.2f}     {h['itemknn']:.3f} {h['tt_a0']:.3f} {h['adj']:.3f}")
    print(f"\nExposure vs demand (total variation distance, lower = exposure follows what users watch): item-kNN {tvd['itemknn']:.3f} | two-tower {tvd['tt_a0']:.3f} | two-tower alpha {a.alpha:g} {tvd['adj']:.3f}")
    print(f"Exposure vs catalogue share (lower = each cluster gets slots in proportion to its size): item-kNN {tvdc['itemknn']:.3f} | two-tower {tvdc['tt_a0']:.3f} | two-tower alpha {a.alpha:g} {tvdc['adj']:.3f}")

    ok = [r for r in rows if r["demand_pct"] >= 1.0 and r["users"] >= 100]
    over = max(ok, key=lambda r: r["ratio"]["tt_a0"])
    under = min(ok, key=lambda r: r["ratio"]["tt_a0"])
    gain = max(ok, key=lambda r: r["hit"]["adj"] - r["hit"]["tt_a0"])
    loss = min(ok, key=lambda r: r["hit"]["adj"] - r["hit"]["tt_a0"])
    gap = max(ok, key=lambda r: r["hit"]["itemknn"] - r["hit"]["adj"])
    found = [
        f"Without adjustment the two-tower puts {over['ratio']['tt_a0']:.2f}x the demand share into C{over['cluster']} and {under['ratio']['tt_a0']:.2f}x into C{under['cluster']}; with alpha {a.alpha:g} these become {over['ratio']['adj']:.2f}x and {under['ratio']['adj']:.2f}x.",
        f"The distance of exposure from demand {'falls' if tvd['adj'] < tvd['tt_a0'] else 'rises'} from {tvd['tt_a0']:.3f} to {tvd['adj']:.3f}, and its distance from catalogue share {'falls' if tvdc['adj'] < tvdc['tt_a0'] else 'rises'} from {tvdc['tt_a0']:.3f} to {tvdc['adj']:.3f} (item-kNN: {tvd['itemknn']:.3f} and {tvdc['itemknn']:.3f}).",
        f"hit@10 changes most favourably in C{gain['cluster']} ({gain['hit']['tt_a0']:.3f} to {gain['hit']['adj']:.3f}, {gain['users']} users) and most unfavourably in C{loss['cluster']} ({loss['hit']['tt_a0']:.3f} to {loss['hit']['adj']:.3f}, {loss['users']} users).",
        f"item-kNN's largest lead over the adjusted two-tower is in C{gap['cluster']} ({gap['hit']['itemknn']:.3f} vs {gap['hit']['adj']:.3f}, {gap['users']} users).",
    ]
    print("\n== Findings computed directly from the table ==")
    for f in found:
        print("-", f)

    llm = {"names": "", "summary": "", "unsupported_numbers": []}
    if a.llm.lower() != "none":
        try:
            print(f"\n== LLM output ({a.llm}); suggestions only, check against the tables above ==")
            llm["names"], llm["summary"] = run_llm(a.llm, [name_prompt(rows, words), rewrite_prompt(found)], dev)
            llm["names"] = clean_names(llm["names"])
            llm["unsupported_numbers"] = unsupported_numbers(llm["summary"], "\n".join(found))
            print("Suggested cluster names:")
            print(llm["names"])
            print("\nParagraph rewritten from the findings above:")
            print(llm["summary"])
            print("number check:", "OK, every number appears in the findings" if not llm["unsupported_numbers"] else f"WARNING, numbers not in the findings: {llm['unsupported_numbers']}")
        except Exception as e:
            print(f"LLM step skipped: {type(e).__name__}: {str(e)[:200]}")

    out = {"k": K, "alpha": a.alpha, "tvd_demand": tvd, "tvd_catalog": tvdc, "clusters": rows, "findings": found, "llm_model": a.llm, "llm": llm}
    (a.reports / "day7_clusters.json").write_text(json.dumps(out, indent=2, default=float))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(K)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.27, [r["demand_pct"] for r in rows], 0.27, label="demand", color="#555555")
    ax.bar(x, [r["expo_pct"]["tt_a0"] for r in rows], 0.27, label="two-tower", color="#999999")
    ax.bar(x + 0.27, [r["expo_pct"]["adj"] for r in rows], 0.27, label=f"two-tower, alpha {a.alpha:g}", color="#cccccc", edgecolor="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in range(K)])
    ax.set_ylabel("% of top-10 slots / % of demand")
    ax.set_title("Exposure vs demand by content cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.reports / "cluster_exposure.png", dpi=150)
    print("\nsaved day7_clusters.json and cluster_exposure.png")


if __name__ == "__main__":
    main()
