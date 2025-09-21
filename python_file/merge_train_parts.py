# merge_train_parts.py (빠른 스니펫)
import glob, polars as pl
paths = sorted(glob.glob("./ae_artifacts/train_ae_c0_part*.parquet"))
df = pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")
df.write_parquet("../data/train_ae_all_c0.parquet")
print(df.shape)
