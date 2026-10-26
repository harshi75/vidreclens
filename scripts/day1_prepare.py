import argparse
import json
from pathlib import Path

import polars as pl

from vidreclens.config import MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS, PROCESSED, RAW, REPORTS
from vidreclens.data.clean import clean_interactions
from vidreclens.data.eda import plot_popularity, summarize
from vidreclens.data.load import load_interactions
from vidreclens.data.split import leave_last_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=PROCESSED)
    p.add_argument("--reports", type=Path, default=REPORTS)
    p.add_argument("--min-user", type=int, default=MIN_USER_INTERACTIONS)
    p.add_argument("--min-item", type=int, default=MIN_ITEM_INTERACTIONS)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.reports.mkdir(parents=True, exist_ok=True)

    lf = load_interactions(a.raw)
    n_raw = lf.select(pl.len()).collect().item()
    df, users, items = clean_interactions(lf, a.min_user, a.min_item)
    df = leave_last_out(df)

    df.write_parquet(a.out / "interactions.parquet")
    users.write_parquet(a.out / "users.parquet")
    items.write_parquet(a.out / "items.parquet")

    stats = {"raw_rows": n_raw, **summarize(df)}
    (a.reports / "stats.json").write_text(json.dumps(stats, indent=2))
    plot_popularity(df, a.reports / "popularity_longtail.png")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
