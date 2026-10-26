import numpy as np

KS = (10, 50)


def tie_aware_rank(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    t = scores[np.arange(len(target)), target][:, None]
    greater = (scores > t).sum(1)
    equal = (scores == t).sum(1)
    return 1 + greater + (equal - 1) / 2


def summarize(ranks: np.ndarray, head: np.ndarray, ks=KS) -> dict:
    out = {}
    for k in ks:
        hit = ranks <= k
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.where(hit, 1 / np.log2(ranks + 1), 0).mean())
        out[f"recall@{k}_head"] = float(hit[head].mean())
        out[f"recall@{k}_tail"] = float(hit[~head].mean())
    return out


def evaluate(score_fn, target: np.ndarray, head_items: np.ndarray, batch: int = 2048) -> dict:
    n = len(target)
    ranks = np.empty(n)
    for a in range(0, n, batch):
        idx = np.arange(a, min(a + batch, n))
        ranks[idx] = tie_aware_rank(score_fn(idx), target[idx])
    return summarize(ranks, head_items[target])
