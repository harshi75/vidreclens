import numpy as np
import polars as pl


def padded_history(df: pl.DataFrame, splits, n_users: int, n_items: int, max_len: int) -> np.ndarray:
    d = df.filter(pl.col("split").is_in(list(splits))).sort("uid", "ts", "iid")
    uid = d["uid"].to_numpy()
    iid = d["iid"].to_numpy()
    counts = np.bincount(uid, minlength=n_users)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(len(uid)) - start[uid]
    drop = np.maximum(counts - max_len, 0)[uid]
    keep = pos >= drop
    P = np.full((n_users, max_len), n_items, dtype=np.int64)
    P[uid[keep], (pos - drop)[keep]] = iid[keep]
    return P
