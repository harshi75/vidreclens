import csv
import json
import math
from pathlib import Path

import numpy as np

_ERF = np.vectorize(math.erf, otypes=[np.float64])


class Recommender:
    def __init__(self, bundle_dir):
        d = Path(bundle_dir)
        z = np.load(d / "model.npz")
        meta = json.loads((d / "meta.json").read_text())
        self.V = z["item_vecs"].astype(np.float32)
        self.W1, self.b1, self.W2, self.b2 = (z[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2"))
        self.logpop = z["logpop"].astype(np.float32)
        self.pop = z["pop_counts"]
        self.item_ids = z["item_ids"]
        self.index = {int(v): i for i, v in enumerate(self.item_ids)}
        self.scale = float(meta["scale"])
        self.alpha = float(meta["alpha"])
        self.max_len = int(meta["max_len"])
        with open(d / "items.csv", newline="", encoding="utf-8") as f:
            titles = {int(r["item"]): r["title"] for r in csv.DictReader(f)}
        self.titles = [titles.get(int(v), "") for v in self.item_ids]
        self.by_pop = np.argsort(-self.pop, kind="stable")

    def user_vector(self, idx):
        x = self.V[idx].mean(0)
        h = x @ self.W1.T + self.b1
        h = 0.5 * h * (1.0 + _ERF(h / math.sqrt(2.0)))
        u = (h @ self.W2.T + self.b2).astype(np.float32)
        return u / max(float(np.linalg.norm(u)), 1e-12)

    def item(self, item_id):
        i = self.index.get(int(item_id))
        if i is None:
            return None
        return {"item_id": int(item_id), "title": self.titles[i], "past_interactions": int(self.pop[i])}

    def recommend(self, history, k=10, alpha=None):
        a = self.alpha if alpha is None else float(alpha)
        known, unknown, seen = [], [], set()
        for h in history:
            if h in seen:
                continue
            seen.add(h)
            i = self.index.get(int(h))
            (known if i is not None else unknown).append(i if i is not None else int(h))
        known = known[-self.max_len :]
        n = len(self.item_ids)
        if not known:
            top = self.by_pop[:k]
            scores = self.pop[top].astype(float)
            cold = True
        else:
            u = self.user_vector(known)
            s = self.scale * (self.V @ u) - a * self.logpop
            s[known] = -np.inf
            kk = min(k, n - len(known))
            top = np.argpartition(-s, kk - 1)[:kk]
            top = top[np.argsort(-s[top])]
            scores = s[top]
            cold = False
        items = [{"item_id": int(self.item_ids[j]), "title": self.titles[j], "score": float(sc)} for j, sc in zip(top, scores)]
        return {"items": items, "unknown_items": unknown, "cold_start": cold, "alpha": a}
