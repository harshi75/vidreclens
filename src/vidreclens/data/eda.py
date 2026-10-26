from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    c = np.cumsum(x)
    return float((len(x) + 1 - 2 * c.sum() / c[-1]) / len(x))


def item_counts(df: pl.DataFrame) -> np.ndarray:
    return df.group_by("iid").len()["len"].to_numpy()


def summarize(df: pl.DataFrame) -> dict:
    ic = item_counts(df)
    uc = df.group_by("uid").len()["len"].to_numpy()
    srt = np.sort(ic)[::-1]
    k = max(1, int(0.2 * len(srt)))
    n_users, n_items, n = len(uc), len(ic), df.height
    return {
        "users": n_users,
        "items": n_items,
        "interactions": n,
        "sparsity": 1 - n / (n_users * n_items),
        "seq_len_p50": float(np.percentile(uc, 50)),
        "seq_len_p90": float(np.percentile(uc, 90)),
        "seq_len_max": int(uc.max()),
        "item_gini": gini(ic),
        "top20pct_items_share": float(srt[:k].sum() / srt.sum()),
        "span_days": (df["ts"].max() - df["ts"].min()) / 86400,
    }


def plot_popularity(df: pl.DataFrame, path: Path) -> None:
    srt = np.sort(item_counts(df))[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(np.arange(1, len(srt) + 1), srt)
    ax.set_xlabel("item rank")
    ax.set_ylabel("interactions")
    ax.set_title("Item popularity (long tail)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
