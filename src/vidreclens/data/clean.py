import polars as pl


def kcore(df: pl.DataFrame, min_user: int, min_item: int) -> pl.DataFrame:
    while True:
        n = df.height
        df = df.filter(pl.len().over("item") >= min_item)
        df = df.filter(pl.len().over("user") >= min_user)
        if df.height == n:
            return df


def reindex(df: pl.DataFrame):
    users = df.select("user").unique().sort("user").with_row_index("uid")
    items = df.select("item").unique().sort("item").with_row_index("iid")
    users = users.with_columns(pl.col("uid").cast(pl.Int64))
    items = items.with_columns(pl.col("iid").cast(pl.Int64))
    out = df.join(users, on="user").join(items, on="item").select("uid", "iid", "ts")
    return out.sort("uid", "ts", "iid"), users, items


def clean_interactions(lf: pl.LazyFrame, min_user: int, min_item: int):
    df = (
        lf.drop_nulls()
        .sort("ts", "user", "item")
        .unique(subset=["user", "item"], keep="first", maintain_order=True)
        .collect()
    )
    return reindex(kcore(df, min_user, min_item))
