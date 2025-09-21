# dataloader_streaming_full.py
# - Tabular: 모든 컬럼 "숫자 특성"으로 처리 (라벨인코딩 없음)
#   * object형 숫자문자: float→round→Int64
#   * l_feat_*: float→round→Int64
#   * feat_*, history_*: float 그대로
#   * hour / day_of_week: mode로 NA 채움 → sin/cos 파생 → 원본 drop
#   * 모든 숫자: train 중앙값으로 NA 채움 → (train 기준) 표준화(mean/std)
# - Seq: *_seq_ids.parquet 스트리밍 로딩, collate에서 PAD로 패딩 (+ tail truncate 옵션)
# - 대용량 안전: IterableDataset + chunk streaming (+ 멀티워커 분할)
# - 경고 제거: tab 텐서는 np.asarray→torch.from_numpy
# - 마스크는 bool로 만들어 메모리 절감

import json, math
import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Tuple, Dict

# ===== 경로 =====
TRAIN_NOSEQ = "../data/train_noseq.parquet"
TEST_NOSEQ  = "../data/test_noseq.parquet"
TRAIN_SEQID = "../data/train_seq_ids.parquet"
TEST_SEQID  = "../data/test_seq_ids.parquet"
VOCAB_JSON  = "../data/seq_vocab_full.json"

# ===== 설정 =====
BATCH_SIZE     = 512
NUM_WORKERS    = 4          # IterableDataset은 0부터, 필요시 2~4로 ↑ (아래 __iter__ 분할 코드 포함)
PERSISTENT_WKR = True
PIN_MEMORY     = True
CHUNK_ROWS     = 1_000_000    # 청크 크기 (메모리/IO에 맞춰 조절)
BATCH_MAX_LEN  = 512        # 0이면 자르지 않음. 512~1024 추천
USE_CYCLICAL   = True       # hour/day_of_week → sin/cos
DEBUG          = True

def dprint(*a, **k):
    if DEBUG: print(*a, **k)

# ---------- Vocab ----------
def load_vocab_meta(path: str) -> Tuple[int, int]:
    raw = json.load(open(path, "r", encoding="utf-8"))
    PAD = int(raw.get("<PAD>", 0))
    UNK = int(raw.get("<UNK>", 1))
    vocab_size = max(raw.values()) + 1 if raw else 0
    dprint(f"[vocab] PAD={PAD}, UNK={UNK}, vocab≈{vocab_size}")
    return PAD, UNK

# ---------- 캐스팅 유틸 ----------
def cast_prefix_to_int(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """l_feat_* 같은 접두어 컬럼을 float→round→Int64로 강제 캐스팅."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols: return df
    exprs = [pl.col(c).cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False).alias(c) for c in cols]
    return df.with_columns(exprs)

def cast_object_numeric_to_int(df: pl.DataFrame, obj_cols: List[str], skip: Optional[set] = None) -> pl.DataFrame:
    """object형(문자)인데 내용이 숫자면 float→round→Int64. (ID 등 스킵 가능)"""
    out = df
    skip = skip or set()
    for c in obj_cols:
        if c in skip:  # ★ ID 등은 건드리지 않음
            continue
        out = out.with_columns(pl.col(c).cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False).alias(c))
    return out

# ---------- 전처리 파라미터(중앙값/모드/평균/표준편차) ----------
class TabStats:
    """
    train에서 중앙값/모드/표준화 파라미터(평균/표준편차)를 집계로 계산.
    표준편차는 population 기준: std = sqrt(E[x^2] - (E[x])^2).
    """
    def __init__(self):
        self.num_cols: List[str] = []      # hour/day_of_week 제외한 숫자 원본 컬럼
        self.used_cyc: List[str] = []      # ['hour','day_of_week']
        self.medians: Dict[str, float] = {}
        self.mean: Dict[str, float] = {}   # sin/cos 포함 agg 컬럼에 대한 mean
        self.std: Dict[str, float]  = {}   # sin/cos 포함 agg 컬럼에 대한 std (0→1로 보정)
        self.mode_hour: Optional[int] = None
        self.mode_dow:  Optional[int] = None

    def fit_from_train(self, train_path: str):
        # 간단히 스키마 파악
        df_head = pl.read_parquet(train_path, n_rows=4)
        cols = df_head.columns

        # object 후보
        obj_like_cols = [c for c,t in df_head.schema.items() if str(t).lower() in ("utf8","string")]

        # 숫자 후보 (label/ID/seq 제외)
        ignore = {"clicked", "seq", "ID"}
        num_cols = []
        for c, t in df_head.schema.items():
            tl = str(t).lower()
            if c in ignore: continue
            if "int" in tl or "float" in tl or c in ("hour","day_of_week") or c.startswith("l_feat_"):
                num_cols.append(c)

        # hour/day_of_week mode (Lazy)
        lf_mode = pl.scan_parquet(train_path)
        # mode()가 여러 값을 내면 첫 값을 사용
        mode_hour = pl.when(pl.col("hour").is_not_null()).then(pl.col("hour")).otherwise(None)
        mode_dow  = pl.when(pl.col("day_of_week").is_not_null()).then(pl.col("day_of_week")).otherwise(None)
        lf_mode = lf_mode.select([
            pl.col("hour").drop_nulls().mode().alias("mode_hour") if "hour" in cols else pl.lit(None).alias("mode_hour"),
            pl.col("day_of_week").drop_nulls().mode().alias("mode_dow") if "day_of_week" in cols else pl.lit(None).alias("mode_dow"),
        ])
        modes = lf_mode.collect()
        self.mode_hour = int(modes["mode_hour"][0]) if "hour" in cols and modes["mode_hour"][0] is not None else 0
        self.mode_dow  = int(modes["mode_dow"][0])  if "day_of_week" in cols and modes["mode_dow"][0] is not None else 0
        self.used_cyc  = [c for c in ("hour", "day_of_week") if c in cols and USE_CYCLICAL]

        # 중앙값 계산(주기형 제외)
        lf_med = pl.scan_parquet(train_path)
        # l_feat_* → int
        lf_med = lf_med.with_columns([pl.col(c).cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False)
                                      for c in cols if c.startswith("l_feat_")])
        # object 숫자문자 → int (ID 제외)
        for c in obj_like_cols:
            if c == "ID": continue
            lf_med = lf_med.with_columns(pl.col(c).cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False))

        base_num_cols = [c for c in num_cols if c not in ("hour", "day_of_week")]
        lf_med = lf_med.select([pl.col(c).median().alias(c) for c in base_num_cols])
        meds_df = lf_med.collect()
        self.medians = {c: float(meds_df[c][0]) for c in base_num_cols}
        self.num_cols = base_num_cols

        # 표준화 mean/std 계산 (sin/cos 포함)
        lf_std = pl.scan_parquet(train_path)
        lf_std = lf_std.with_columns([pl.col(c).cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False)
                                      for c in cols if c.startswith("l_feat_")])
        for c in obj_like_cols:
            if c == "ID": continue
            lf_std = lf_std.with_columns(pl.col(c).cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False))

        agg_cols = list(self.num_cols)
        if "hour" in self.used_cyc:
            lf_std = lf_std.with_columns(pl.col("hour").fill_null(self.mode_hour).cast(pl.Int64, strict=False))
            lf_std = lf_std.with_columns([
                (pl.col("hour") * (2*np.pi/24)).sin().alias("hour_sin"),
                (pl.col("hour") * (2*np.pi/24)).cos().alias("hour_cos"),
            ]).drop("hour")
            agg_cols += ["hour_sin", "hour_cos"]
        if "day_of_week" in self.used_cyc:
            lf_std = lf_std.with_columns(pl.col("day_of_week").fill_null(self.mode_dow).cast(pl.Int64, strict=False))
            lf_std = lf_std.with_columns([
                (pl.col("day_of_week") * (2*np.pi/7)).sin().alias("day_of_week_sin"),
                (pl.col("day_of_week") * (2*np.pi/7)).cos().alias("day_of_week_cos"),
            ]).drop("day_of_week")
            agg_cols += ["day_of_week_sin", "day_of_week_cos"]

        # 중앙값 채움
        for c in self.num_cols:
            lf_std = lf_std.with_columns(pl.col(c).fill_null(self.medians[c]).alias(c))

        # mean, mean2 집계 → std = sqrt(mean2 - mean^2)
        lf_std = lf_std.select(
            [pl.col(c).mean().alias(f"{c}__mean") for c in agg_cols] +
            [(pl.col(c) * pl.col(c)).mean().alias(f"{c}__mean2") for c in agg_cols]
        )
        stats_df = lf_std.collect()

        for c in agg_cols:
            m  = float(stats_df[f"{c}__mean"][0])
            m2 = float(stats_df[f"{c}__mean2"][0])
            var = max(m2 - m*m, 1e-12)
            self.mean[c] = m
            self.std[c]  = var ** 0.5

        dprint(f"[stats] num_cols={len(self.num_cols)}, cyc={self.used_cyc}, agg_cols={len(agg_cols)}")
        dprint(f"[stats] example mean/std: {[(k, round(self.mean[k],3), round(self.std[k],3)) for k in list(self.mean.keys())[:5]]}")

    def transform_chunk(self, df: pl.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], Optional[np.ndarray]]:
        """
        df: 같은 오프셋의 tabular 청크
        반환: X_num(float32), y(optional), 사용 컬럼 리스트, ID(optional)
        """
        # l_feat_* → int, object 숫자문자 → int (ID 제외)
        df = cast_prefix_to_int(df, "l_feat_")
        obj_cols = [c for c,t in df.schema.items() if str(t).lower() in ("utf8","string")]
        df = cast_object_numeric_to_int(df, obj_cols, skip={"ID"})

        # 주기형 채움→sin/cos
        num_used = list(self.num_cols)
        if "hour" in self.used_cyc and "hour" in df.columns:
            df = df.with_columns(pl.col("hour").fill_null(self.mode_hour).cast(pl.Int64, strict=False))
            df = df.with_columns([
                (pl.col("hour") * (2*np.pi/24)).sin().alias("hour_sin"),
                (pl.col("hour") * (2*np.pi/24)).cos().alias("hour_cos"),
            ]).drop("hour")
            num_used += ["hour_sin","hour_cos"]
        if "day_of_week" in self.used_cyc and "day_of_week" in df.columns:
            df = df.with_columns(pl.col("day_of_week").fill_null(self.mode_dow).cast(pl.Int64, strict=False))
            df = df.with_columns([
                (pl.col("day_of_week") * (2*np.pi/7)).sin().alias("day_of_week_sin"),
                (pl.col("day_of_week") * (2*np.pi/7)).cos().alias("day_of_week_cos"),
            ]).drop("day_of_week")
            num_used += ["day_of_week_sin","day_of_week_cos"]

        # 중앙값 채움 (원본 숫자 컬럼만)
        for c in self.num_cols:
            df = df.with_columns(pl.col(c).fill_null(self.medians[c]).alias(c))

        # 넘파이 추출 → 표준화
        X = df.select(num_used).to_numpy().astype(np.float32) if num_used else np.zeros((df.height,0), np.float32)
        if X.shape[1] > 0:
            mu  = np.array([self.mean[c] for c in num_used], dtype=np.float32)
            sig = np.array([self.std[c]  for c in num_used], dtype=np.float32)
            X = (X - mu) / sig

        y = None
        if "clicked" in df.columns:
            y = df["clicked"].to_numpy().astype(np.int64)

        ids = None
        if "ID" in df.columns:
            ids = df["ID"].to_numpy()  # 문자열/정수 그대로

        return X, y, num_used, ids

# ---------- IterableDataset ----------
class SeqTabIterableDataset(IterableDataset):
    """
    tabular/seq를 동일 오프셋의 청크로 읽어 sample 단위로 yield.
    with_label=True면 label 포함, include_id=True면 ID 포함.
    """
    def __init__(self, tab_path: str, seq_path: str, stats: TabStats,
                 chunk_rows: int = CHUNK_ROWS, with_label: bool = True, include_id: bool = False):
        super().__init__()
        self.tab_path   = tab_path
        self.seq_path   = seq_path
        self.stats      = stats
        self.chunk_rows = chunk_rows
        self.with_label = with_label
        self.include_id = include_id
        # 총 행수
        self.N = int(pl.scan_parquet(tab_path).select(pl.len()).collect().item())

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            start_chunk, step = 0, 1
            wid = "main"
        else:
            start_chunk = info.id
            step = info.num_workers
            wid = f"w{info.id}"

        total_chunks = math.ceil(self.N / self.chunk_rows)
        for chunk_idx in range(start_chunk, total_chunks, step):
            start_row = chunk_idx * self.chunk_rows
            length = min(self.chunk_rows, self.N - start_row)
            if length <= 0:
                break
            if DEBUG:
                print(f"[stream {wid}] rows {start_row:,}..{start_row+length-1:,}")

            # tab 청크
            df_tab = pl.scan_parquet(self.tab_path).slice(start_row, length).collect()
            X, y, _, ids = self.stats.transform_chunk(df_tab)

            # seq 청크
            df_seq = (pl.scan_parquet(self.seq_path)
                        .slice(start_row, length)
                        .select(["seq_ids","len"])
                        .collect())
            seqs = df_seq["seq_ids"].to_list()
            lens = df_seq["len"].to_numpy().astype(np.int64)

            assert X.shape[0] == len(seqs) == len(lens)

            for i in range(X.shape[0]):
                s = seqs[i] if len(seqs[i]) > 0 else [1]  # 빈 seq 안전장치: UNK(=1)
                if BATCH_MAX_LEN and len(s) > BATCH_MAX_LEN:
                    s = s[-BATCH_MAX_LEN:]  # 최근 L개만
                item = {
                    "tab_num": X[i],
                    "seq":     s,
                    "length":  min(int(lens[i]) if lens[i] > 0 else 1, len(s))
                }
                if self.with_label:
                    item["label"] = int(y[i])
                if self.include_id and ids is not None:
                    item["id"] = ids[i]
                yield item

# ---------- collate ----------
def make_collate(pad_id: int, unk_id: int):
    def collate(batch):
        # tab_num: 리스트→np.asarray→from_numpy (빠르고 경고 없음)
        X_np = np.asarray([b["tab_num"] for b in batch], dtype=np.float32)
        X = torch.from_numpy(X_np)

        # seq: 패딩(+빈 시퀀스/길이 교정은 IterableDataset에서)
        seq_tensors = [torch.tensor(b["seq"], dtype=torch.long) for b in batch]
        padded = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_id)
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)

        out = {
            "tab_num":   X,
            "seq":       padded,
            "lengths":   lengths,
            "attn_mask": padded.ne(pad_id)  # bool mask → 메모리 절약
        }
        if "label" in batch[0]:
            out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        if "id" in batch[0]:
            out["id"] = [b["id"] for b in batch]
        return out
    return collate

# ---------- 빌더 ----------
def build_stream_loaders():
    PAD, UNK = load_vocab_meta(VOCAB_JSON)

    # 1) 통계(중앙값/모드/mean/std) 계산 — 메모리 적게 사용
    stats = TabStats()
    stats.fit_from_train(TRAIN_NOSEQ)

    # 2) IterableDataset 구성
    ds_tr = SeqTabIterableDataset(TRAIN_NOSEQ, TRAIN_SEQID, stats, with_label=True,  include_id=False)
    ds_te = SeqTabIterableDataset(TEST_NOSEQ,  TEST_SEQID,  stats, with_label=False, include_id=True)

    collate = make_collate(PAD, UNK)
    dl_tr = DataLoader(
        ds_tr, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate, persistent_workers=(NUM_WORKERS>0 and PERSISTENT_WKR),
    )
    dl_te = DataLoader(
        ds_te, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate, persistent_workers=(NUM_WORKERS>0 and PERSISTENT_WKR),
    )

    # 간단 체크
    it = iter(dl_tr)
    b = next(it)
    print(f"[check] train batch: tab={b['tab_num'].shape}, seq={b['seq'].shape}, y={b['label'].shape}")
    it2 = iter(dl_te)
    bt = next(it2)
    print(f"[check] test  batch: tab={bt['tab_num'].shape}, seq={bt['seq'].shape}, id-sample={bt.get('id', [])[:3]}")
    return dl_tr, dl_te, stats

if __name__ == "__main__":
    dl_tr, dl_te, stats = build_stream_loaders()
    print("[done] streaming dataloaders ready.")
