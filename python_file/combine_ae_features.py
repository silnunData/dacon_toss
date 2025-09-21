# combine_ae_features_from_all.py
# - 입력:
#   ../data/train_ae_all_c0.parquet, ../data/train_ae_all_c1.parquet
#   ../data/test_ae_all_c0.parquet,  ../data/test_ae_all_c1.parquet
# - 출력:
#   ../data/train_ae_derived.parquet
#   ../data/test_ae_derived.parquet

import polars as pl
import numpy as np

TRAIN_C0 = "../data/train_ae_all_c0.parquet"
TRAIN_C1 = "../data/train_ae_all_c1.parquet"
TEST_C0  = "../data/test_ae_all_c0.parquet"
TEST_C1  = "../data/test_ae_all_c1.parquet"

OUT_TRAIN = "../data/train_ae_derived.parquet"
OUT_TEST  = "../data/test_ae_derived.parquet"

EPS = 1e-6

def build_train():
    # train에는 ID가 없으므로 row 순서로 맞춘다 → row_idx 부여 후 join
    df0 = pl.read_parquet(TRAIN_C0).with_row_index("row_idx")
    df1 = pl.read_parquet(TRAIN_C1).with_row_index("row_idx")

    if df0.height != df1.height:
        raise RuntimeError(f"row mismatch: c0={df0.height:,}, c1={df1.height:,}")

    df0 = df0.rename({c: f"{c}_c0" for c in df0.columns if c != "row_idx"})
    df1 = df1.rename({c: f"{c}_c1" for c in df1.columns if c != "row_idx"})

    df = df0.join(df1, on="row_idx", how="inner")

    # 파생 피처: err_diff/ratio/logratio
    df = df.with_columns([
        (pl.col("recon_error_c0") - pl.col("recon_error_c1")).alias("err_diff"),
        (pl.col("recon_error_c0") / (pl.col("recon_error_c1") + EPS)).alias("err_ratio"),
        ((pl.col("recon_error_c0") + EPS).log() - (pl.col("recon_error_c1") + EPS).log())
            .alias("err_logratio"),
    ])

    # z 파생: z_diff_i, z_abs_i, z_cos
    z0 = [c for c in df.columns if c.startswith("z_") and c.endswith("_c0")]
    z1 = [c for c in df.columns if c.startswith("z_") and c.endswith("_c1")]
    z0_sorted = sorted(z0, key=lambda x: int(x.split("_")[1]))
    z1_sorted = sorted(z1, key=lambda x: int(x.split("_")[1]))
    assert len(z0_sorted) == len(z1_sorted)
    d = len(z0_sorted)

    num = den0 = den1 = None
    for i in range(d):
        c0 = pl.col(z0_sorted[i])
        c1 = pl.col(z1_sorted[i])
        num  = c0 * c1 if num  is None else num  + c0 * c1
        den0 = c0 * c0 if den0 is None else den0 + c0 * c0
        den1 = c1 * c1 if den1 is None else den1 + c1 * c1
        df = df.with_columns([
            (pl.col(z1_sorted[i]) - pl.col(z0_sorted[i])).alias(f"z_diff_{i}"),
            (pl.col(z1_sorted[i]) - pl.col(z0_sorted[i])).abs().alias(f"z_abs_{i}"),
        ])
    df = df.with_columns((num / (den0.sqrt() * den1.sqrt() + EPS)).alias("z_cos"))

    df.write_parquet(OUT_TRAIN)
    print(f"[train] saved {OUT_TRAIN} rows={df.height:,}, cols={len(df.columns)}")

def build_test():
    # test에는 ID가 있으므로 ID로 조인
    df0 = pl.read_parquet(TEST_C0)
    df1 = pl.read_parquet(TEST_C1)
    if "ID" not in df0.columns or "ID" not in df1.columns:
        raise RuntimeError("Test AE files must include 'ID' column.")

    df0 = df0.rename({c: f"{c}_c0" for c in df0.columns if c != "ID"})
    df1 = df1.rename({c: f"{c}_c1" for c in df1.columns if c != "ID"})

    df = df0.join(df1, on="ID", how="inner")

    df = df.with_columns([
        (pl.col("recon_error_c0") - pl.col("recon_error_c1")).alias("err_diff"),
        (pl.col("recon_error_c0") / (pl.col("recon_error_c1") + EPS)).alias("err_ratio"),
        ((pl.col("recon_error_c0") + EPS).log() - (pl.col("recon_error_c1") + EPS).log())
            .alias("err_logratio"),
    ])

    z0 = [c for c in df.columns if c.startswith("z_") and c.endswith("_c0")]
    z1 = [c for c in df.columns if c.startswith("z_") and c.endswith("_c1")]
    z0_sorted = sorted(z0, key=lambda x: int(x.split("_")[1]))
    z1_sorted = sorted(z1, key=lambda x: int(x.split("_")[1]))
    d = len(z0_sorted)

    num = den0 = den1 = None
    for i in range(d):
        c0 = pl.col(z0_sorted[i])
        c1 = pl.col(z1_sorted[i])
        num  = c0 * c1 if num  is None else num  + c0 * c1
        den0 = c0 * c0 if den0 is None else den0 + c0 * c0
        den1 = c1 * c1 if den1 is None else den1 + c1 * c1
        df = df.with_columns([
            (pl.col(z1_sorted[i]) - pl.col(z0_sorted[i])).alias(f"z_diff_{i}"),
            (pl.col(z1_sorted[i]) - pl.col(z0_sorted[i])).abs().alias(f"z_abs_{i}"),
        ])
    df = df.with_columns((num / (den0.sqrt() * den1.sqrt() + EPS)).alias("z_cos"))

    df.write_parquet(OUT_TEST)
    print(f"[test] saved {OUT_TEST} rows={df.height:,}, cols={len(df.columns)}")

if __name__ == "__main__":
    build_train()
    build_test()
