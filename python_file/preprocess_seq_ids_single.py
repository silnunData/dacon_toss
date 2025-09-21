# make_seq_ids_progress.py
import os, json, math
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# ===== 경로 =====
VOCAB_JSON    = "../data/seq_vocab_full.json"
TRAIN_SEQ_IN  = "../data/train_seqlist_tot.parquet"  # 'seq' list[int]
TEST_SEQ_IN   = "../data/test_seqlist.parquet"           # 'seq' list[int]
TEST_NOSEQ_IN = "../data/test_noseq.parquet"         # 'ID' 한 컬럼

TRAIN_SEQ_OUT = "../data/train_seq_ids.parquet"
TEST_SEQ_OUT  = "../data/test_seq_ids.parquet"

# ===== 옵션 =====
MAX_LEN   = 0            # 0=전체, >0 이면 뒤에서 MAX_LEN개만
CHUNK     = 1_000_000    # 청크 행수
KEEP_ID   = True         # test에 ID 붙이기
DEBUG     = True

def dprint(*a, **k):
    if DEBUG: print(*a, **k)

def load_vocab(path: str):
    print(f"[vocab] loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "<UNK>" not in raw:
        raise ValueError("'<UNK>' missing in vocab")
    UNK = int(raw["<UNK>"])
    stoi = {}
    for k, v in raw.items():
        if k in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"): continue
        stoi[int(k)] = int(v)
    print(f"[vocab] UNK={UNK}, mapped tokens={len(stoi):,}")
    return stoi, UNK

def make_map_expr(stoi: dict, UNK: int):
    elem = pl.element()
    if hasattr(elem, "replace_strict"):
        dprint("[map] replace_strict")
        return elem.replace_strict(stoi, default=UNK)
    if hasattr(elem, "replace"):
        dprint("[map] replace (fallback)")
        try:
            return elem.replace(stoi, default=UNK)
        except TypeError:
            dprint("[map] replace w/o default → fill_null(UNK)")
            return elem.replace(stoi).fill_null(UNK)
    dprint("[map] apply (slow fallback)")
    return elem.apply(lambda x: stoi.get(x, UNK))

def trim_expr(expr: pl.Expr, max_len: int):
    return expr.list.tail(max_len) if (max_len and max_len > 0) else expr

def total_rows(parquet_path: str) -> int:
    return int(pl.scan_parquet(parquet_path).select(pl.len()).collect().item())

def ensure_parent(path: str):
    p = os.path.dirname(path)
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def process_split(seq_in: str, out_path: str, stoi: dict, UNK: int,
                  label: str, keep_id: bool=False, id_path: str|None=None):
    print(f"[{label}] input: {seq_in}")
    n = total_rows(seq_in)
    print(f"[{label}] total rows: {n:,}")
    steps = math.ceil(n / CHUNK)
    map_expr = make_map_expr(stoi, UNK)

    writer = None
    ensure_parent(out_path)

    for step in range(steps):
        start = step * CHUNK
        length = min(CHUNK, n - start)
        print(f"[{label}] chunk {step+1}/{steps}  rows {start:,}..{start+length-1:,}")

        # 1) seq slice
        lf = (
            pl.scan_parquet(seq_in)
              .slice(start, length)
              .select(pl.col("seq"))
              .with_columns(trim_expr(pl.col("seq"), MAX_LEN).alias("seq_trim"))
              .with_columns(pl.col("seq_trim").list.eval(map_expr).alias("seq_ids"))
              .with_columns(pl.col("seq_ids").list.len().cast(pl.Int32).alias("len"))
              .select(["seq_ids","len"])
        )
        df = lf.collect()  # Eager materialize this chunk
        dprint(f"[{label}]  - mapped df shape: {df.shape}")

        # 2) (test) attach ID
        if keep_id:
            if id_path is None:
                raise ValueError("id_path required when keep_id=True")
            lfid = (
                pl.scan_parquet(id_path)
                  .slice(start, length)
                  .select(pl.col("ID"))
            )
            dfid = lfid.collect()
            if dfid.height != df.height:
                raise RuntimeError(f"[{label}] ID rows({dfid.height}) != seq rows({df.height}) at chunk {step}")
            df = df.with_columns(dfid["ID"]).select(["ID","seq_ids","len"])
            dprint(f"[{label}]  - with ID shape: {df.shape}")

        # 3) append to single parquet (arrow writer)
        table = df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema=table.schema, compression="zstd")
        writer.write_table(table)

        # 4) small probe
        if step == 0:
            dprint(f"[{label}]  - head(2):\n{df.head(2)}")

    if writer is not None:
        writer.close()
    print(f"[{label}] ✅ saved: {out_path}")

def main():
    stoi, UNK = load_vocab(VOCAB_JSON)

    # TRAIN
    process_split(TRAIN_SEQ_IN, TRAIN_SEQ_OUT, stoi, UNK, label="train", keep_id=False)

    # TEST
    process_split(TEST_SEQ_IN, TEST_SEQ_OUT, stoi, UNK, label="test",
                  keep_id=KEEP_ID, id_path=(TEST_NOSEQ_IN if KEEP_ID else None))

    # verify outputs
    for p in (TRAIN_SEQ_OUT, TEST_SEQ_OUT):
        if os.path.exists(p):
            df = pl.read_parquet(p).head(2)
            print(f"[verify] {p} head(2):\n{df}")
        else:
            print(f"[verify] missing: {p}")

if __name__ == "__main__":
    main()
