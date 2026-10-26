import numpy as np
import polars as pl
import scipy.sparse as sp


def history_matrix(df: pl.DataFrame, n_users: int, n_items: int, splits) -> sp.csr_matrix:
    d = df.filter(pl.col("split").is_in(list(splits)))
    data = np.ones(d.height, dtype=np.float32)
    return sp.csr_matrix((data, (d["uid"].to_numpy(), d["iid"].to_numpy())), shape=(n_users, n_items))


def mask_history(s: np.ndarray, H: sp.csr_matrix, idx: np.ndarray) -> np.ndarray:
    sub = H[idx]
    rows = np.repeat(np.arange(len(idx)), np.diff(sub.indptr))
    s[rows, sub.indices] = -np.inf
    return s


def popularity(H):
    pop = np.asarray(H.sum(0)).ravel().astype(np.float32)

    def fn(idx):
        return mask_history(np.tile(pop, (len(idx), 1)), H, idx)

    return fn


def content_profile(H, E):
    deg = np.asarray(H.sum(1)).ravel().clip(1)
    U = np.asarray(H @ E) / deg[:, None]
    U /= np.linalg.norm(U, axis=1, keepdims=True).clip(1e-8)

    def fn(idx):
        return mask_history(U[idx] @ E.T, H, idx)

    return fn


def itemknn(H):
    G = (H.T @ H).toarray().astype(np.float32)
    d = np.sqrt(np.diag(G)).clip(1e-8)
    G /= d[:, None]
    G /= d[None, :]
    np.fill_diagonal(G, 0)

    def fn(idx):
        return mask_history(np.asarray(H[idx] @ G), H, idx)

    return fn
