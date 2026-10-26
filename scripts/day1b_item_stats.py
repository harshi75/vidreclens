import argparse
from pathlib import Path

import polars as pl

from vidreclens.config import PROCESSED, RAW
from vidreclens.data.load import load_item_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=PROCESSED)
    a = p.parse_args()

    items = pl.read_parquet(a.out / "items.parquet")
    stats = (
        items.join(load_item_stats(a.raw), on="item", how="left")
        .with_columns(
            (pl.col("likes") / pl.col("views")).alias("like_rate"),
            pl.col("views").log1p().alias("log_views"),
        )
        .select("iid", "item", "likes", "views", "like_rate", "log_views")
    )
    stats.write_parquet(a.out / "item_stats.parquet")
    print(stats.describe())
    print("matched:", stats["views"].is_not_null().mean())


if __name__ == "__main__":
    main()
