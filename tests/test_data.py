import polars as pl

from vidreclens.data.clean import clean_interactions, kcore
from vidreclens.data.eda import gini, summarize
from vidreclens.data.split import leave_last_out
import numpy as np


def make_raw():
    rows = []
    for u in range(1, 21):
        for j in range(8):
            rows.append((u, 100 + (u + j) % 10, 1000 + j))
    rows += [(1, 100, 5000), (1, 100, 5001)]
    rows += [(99, 777, 1), (99, 778, 2)]
    return pl.DataFrame(rows, schema=["user", "item", "ts"], orient="row").lazy()


def test_clean_dedup_and_kcore():
    df, users, items = clean_interactions(make_raw(), 5, 5)
    assert df.group_by("uid").len()["len"].min() >= 5
    assert df.group_by("iid").len()["len"].min() >= 5
    assert df.select(pl.struct("uid", "iid").is_duplicated().any()).item() is False
    assert 99 not in users["user"].to_list()
    assert sorted(df["uid"].unique().to_list()) == list(range(users.height))
    assert sorted(df["iid"].unique().to_list()) == list(range(items.height))


def test_split_is_temporal_and_complete():
    df, _, _ = clean_interactions(make_raw(), 5, 5)
    sp = leave_last_out(df)
    per_user = sp.group_by("uid").agg(
        (pl.col("split") == "test").sum().alias("t"),
        (pl.col("split") == "val").sum().alias("v"),
    )
    assert per_user["t"].to_list() == [1] * per_user.height
    assert per_user["v"].to_list() == [1] * per_user.height
    mx = sp.filter(pl.col("split") == "train").group_by("uid").agg(pl.col("ts").max().alias("m"))
    tmin = sp.filter(pl.col("split") == "test").select("uid", "ts").join(mx, on="uid")
    assert (tmin["ts"] >= tmin["m"]).all()


def test_gini_bounds():
    assert abs(gini(np.ones(100))) < 1e-9
    assert gini(np.array([0] * 99 + [100])) > 0.98


def test_summarize_keys():
    df, _, _ = clean_interactions(make_raw(), 5, 5)
    s = summarize(leave_last_out(df))
    assert 0 < s["sparsity"] < 1 and s["users"] == 20


def test_padded_history_keeps_latest_and_pads():
    from vidreclens.data.history import padded_history

    df = pl.DataFrame(
        {
            "uid": [0, 0, 0, 0, 0, 1, 1, 1],
            "iid": [5, 6, 7, 8, 9, 1, 2, 3],
            "ts": [1, 2, 3, 4, 5, 1, 2, 3],
            "split": ["train"] * 4 + ["val"] + ["train"] * 3,
        }
    )
    P = padded_history(df, ["train"], 3, 10, 3)
    assert P.tolist() == [[6, 7, 8], [1, 2, 3], [10, 10, 10]]
    P2 = padded_history(df, ["train", "val"], 3, 10, 3)
    assert P2[0].tolist() == [7, 8, 9]
