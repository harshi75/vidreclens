import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from vidreclens.serve.engine import Recommender  # noqa: E402

BUNDLE = Path(os.environ.get("VIDRECLENS_BUNDLE", ROOT / "bundle"))

st.set_page_config(page_title="VidRec-Lens", layout="wide")


@st.cache_resource
def load():
    eng = Recommender(BUNDLE)
    head = np.zeros(len(eng.item_ids), dtype=bool)
    head[eng.by_pop[: int(0.2 * len(eng.item_ids))]] = True
    return eng, head


eng, head = load()
has_titles = any(eng.titles)
hist = st.session_state.setdefault("history", [])


def label(i):
    t = eng.titles[i]
    return f"{int(eng.item_ids[i])}: {t[:70]}" if t else f"video {int(eng.item_ids[i])}"


def table(r):
    rows = []
    for rank, it in enumerate(r["items"], 1):
        i = eng.index[it["item_id"]]
        rows.append({"rank": rank, "video": label(i), "past interactions": int(eng.pop[i]), "tail video": "no" if head[i] else "yes", "score": round(it["score"], 2)})
    return pd.DataFrame(rows)


def summary(r):
    idx = [eng.index[it["item_id"]] for it in r["items"]]
    return float(np.mean([not head[i] for i in idx])), float(np.median([eng.pop[i] for i in idx]))


st.title("VidRec-Lens")
st.caption("Content-aware video recommendation on MicroLens-100K. Pick videos a user watched, then see what the model recommends with and without the popularity adjustment.")

with st.sidebar:
    k = st.slider("Recommendations", 5, 20, 10)
    alpha = st.slider("Popularity penalty (alpha)", 0.0, 2.0, float(eng.alpha), 0.05)
    compare = st.checkbox("Compare with no adjustment (alpha 0)", value=True)
    st.markdown("alpha 0 ranks by taste only and favours popular videos. Higher alpha pushes toward less-seen videos. The project's chosen value is 0.75.")

st.subheader("1. Build a watch history")
q = st.text_input("Search by title" if has_titles else "Find a video by its numeric id", "")
matches = []
if q.strip():
    if has_titles:
        matches = [i for i, t in enumerate(eng.titles) if q.lower() in t.lower()][:50]
    elif q.strip().isdigit() and int(q) in eng.index:
        matches = [eng.index[int(q)]]
    if not matches:
        st.caption("No match.")
if matches:
    pick = st.selectbox("Matches", matches, format_func=label)
    if st.button("Add to history"):
        vid = int(eng.item_ids[pick])
        if vid not in hist:
            hist.append(vid)

c1, c2, c3 = st.columns(3)
rng = np.random.default_rng()
if c1.button("Add 5 popular videos"):
    hist.extend(int(eng.item_ids[i]) for i in rng.choice(eng.by_pop[:500], 5, replace=False) if int(eng.item_ids[i]) not in hist)
if c2.button("Add 5 random videos"):
    hist.extend(int(eng.item_ids[i]) for i in rng.choice(len(eng.item_ids), 5, replace=False) if int(eng.item_ids[i]) not in hist)
if c3.button("Clear history"):
    hist.clear()

if hist:
    st.dataframe(pd.DataFrame({"order": range(1, len(hist) + 1), "video": [label(eng.index[v]) for v in hist]}), hide_index=True, width="stretch")
else:
    st.info("History is empty, so the model falls back to the most popular videos.")

st.subheader("2. Recommendations")
base = eng.recommend(hist, k, 0.0)
adj = eng.recommend(hist, k, alpha)
if adj["cold_start"]:
    st.info("No usable history: showing the most popular videos.")
if compare and not adj["cold_start"]:
    a, b = st.columns(2)
    for col, r, name in ((a, base, "alpha 0 (no adjustment)"), (b, adj, f"alpha {alpha:g}")):
        with col:
            tail, med = summary(r)
            st.markdown(f"**{name}**")
            st.caption(f"{tail:.0%} of these are tail videos, median past interactions {med:.0f}")
            st.dataframe(table(r), hide_index=True, width="stretch")
    overlap = len({x["item_id"] for x in base["items"]} & {x["item_id"] for x in adj["items"]})
    st.caption(f"{overlap} of {k} recommendations are the same in both lists.")
else:
    tail, med = summary(adj)
    st.caption(f"{tail:.0%} of these are tail videos, median past interactions {med:.0f}")
    st.dataframe(table(adj), hide_index=True, width="stretch")

with st.expander("About this demo"):
    st.markdown(
        "A two-tower model (learned ID embedding plus cover, title and video features) trained on MicroLens-100K. "
        "On the held-out last item per user it ties item-kNN on Recall@10 (0.0581 vs 0.0586) and beats it at Recall@50. "
        "The popularity adjustment raised tail-video recall from 0.0321 to 0.0460 and catalogue coverage from 79% to 97%. "
        "The tail is the 80% of videos outside the top 20% by training popularity. "
        "This demo runs the same numpy engine as the API and loads a small precomputed bundle, so it needs no GPU."
    )
