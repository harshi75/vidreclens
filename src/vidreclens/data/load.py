from pathlib import Path

import polars as pl

from vidreclens.config import RAW


def find_pairs_file(root: Path = RAW) -> Path:
    hits = sorted(Path(root).rglob("*pairs.csv"))
    if not hits:
        raise FileNotFoundError(f"no *pairs.csv under {root}")
    return hits[0]


def load_interactions(root: Path = RAW, path: Path | None = None) -> pl.LazyFrame:
    lf = pl.scan_csv(path or find_pairs_file(root))
    cols = lf.collect_schema().names()[:3]
    lf = lf.select(
        pl.col(cols[0]).alias("user"),
        pl.col(cols[1]).alias("item"),
        pl.col(cols[2]).alias("ts"),
    )
    dtype = lf.collect_schema()["ts"]
    raw = pl.col("ts")
    if dtype == pl.String:
        raw = pl.coalesce(
            raw.cast(pl.Int64, strict=False),
            raw.str.to_datetime(strict=False).dt.epoch("s"),
        )
    elif isinstance(dtype, pl.Datetime):
        raw = raw.dt.epoch("s")
    ts = pl.when(raw > 10**11).then(raw // 1000).otherwise(raw)
    return lf.select(
        pl.col("user").cast(pl.Int64, strict=False),
        pl.col("item").cast(pl.Int64, strict=False),
        ts.cast(pl.Int64).alias("ts"),
    )


def load_item_stats(root: Path = RAW, path: Path | None = None) -> pl.DataFrame:
    path = path or next(Path(root).rglob("*likes_and_views.txt"))
    raw = pl.read_csv(
        path, has_header=False, separator="\x1f", new_columns=["line"], infer_schema=False
    )
    g = pl.col("line").str.extract_groups(r"^\s*(\d+)\s+(\d+)\s+(\d+)").alias("g")
    return (
        raw.select(g)
        .unnest("g")
        .rename({"1": "item", "2": "likes", "3": "views"})
        .drop_nulls()
        .with_columns(pl.all().cast(pl.Int64))
        .unique(subset="item", keep="first")
    )
