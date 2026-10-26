import polars as pl


def leave_last_out(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("uid", "ts", "iid").with_columns(
        pl.int_range(pl.len()).over("uid").alias("pos"),
        pl.len().over("uid").alias("n"),
    )
    split = (
        pl.when(pl.col("pos") == pl.col("n") - 1)
        .then(pl.lit("test"))
        .when(pl.col("pos") == pl.col("n") - 2)
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
    )
    return df.with_columns(split.alias("split")).drop("pos", "n")
