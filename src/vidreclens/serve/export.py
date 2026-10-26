import csv
import json
from pathlib import Path

import numpy as np


def export_bundle(model, item_ids, titles, pop_counts, out_dir, alpha=0.75, max_len=50):
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        IR = model.encode_items().cpu().numpy().astype(np.float32)
    lin1, lin2 = model.user_mlp[0], model.user_mlp[3]
    pop_counts = np.asarray(pop_counts, dtype=np.int64)
    np.savez_compressed(
        out / "model.npz",
        item_vecs=IR[:-1],
        W1=lin1.weight.detach().cpu().numpy(),
        b1=lin1.bias.detach().cpu().numpy(),
        W2=lin2.weight.detach().cpu().numpy(),
        b2=lin2.bias.detach().cpu().numpy(),
        logpop=np.log1p(pop_counts).astype(np.float32),
        pop_counts=pop_counts,
        item_ids=np.asarray(item_ids, dtype=np.int64),
    )
    meta = {"scale": float(min(model.log_scale.exp().item(), 100.0)), "alpha": float(alpha), "max_len": int(max_len), "dim": int(IR.shape[1]), "n_items": int(len(item_ids))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    with open(out / "items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "title"])
        for i, t in zip(item_ids, titles):
            w.writerow([int(i), t])
    return out
