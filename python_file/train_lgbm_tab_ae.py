# train_lgbm_tab_ae_memsafe.py
# - 메모리 안전 버전: AE-only / Tab+AE(네거티브 서브샘플링) 스위치
# - F1@0.5, Best-F1 출력
import os, numpy as np, polars as pl, lightgbm as lgb

# ===== Paths =====
TRAIN_AE = "../data/train_ae_derived.parquet"
TEST_AE  = "../data/test_ae_derived.parquet"
TRAIN_NS = "../data/train_noseq.parquet"   # clicked 포함
TEST_NS  = "../data/test_noseq.parquet"    # ID 포함

OUT_DIR  = "./submissions"
SUB_CSV  = os.path.join(OUT_DIR, "submission_lgbm.csv")
FI_CSV   = os.path.join(OUT_DIR, "feature_importance.csv")
MODEL_TXT= os.path.join(OUT_DIR, "lgbme.txt")
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

# ===== Switches =====
USE_TAB = True          # False: AE 파생만 사용(권장, 메모리↓) / True: Tab+AE 함께
NEG_SAMPLE_FRAC = 0.10   # USE_TAB=True일 때만: train의 label=0을 이 비율만 남김(전체=양성+음성샘플)

# ===== Tab preprocessing helpers =====
OBJ_COLS = ["gender", "age_group", "inventory_id", "day_of_week", "hour"]
L_PREFIX = "l_feat_"

def cast_object_int(df: pl.DataFrame) -> pl.DataFrame:
    for c in OBJ_COLS:
        if c in df.columns:
            if df[c].dtype != pl.Int64:
                df = df.with_columns(
                    pl.col(c).cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False)
                )
            mode_val = df.select(pl.col(c)).drop_nulls().get_column(c).mode()
            fill_v = int(mode_val[0]) if len(mode_val) > 0 else 0
            df = df.with_columns(pl.col(c).fill_null(fill_v))
    return df

def cast_l_feats_int(df: pl.DataFrame) -> pl.DataFrame:
    l_cols = [c for c in df.columns if c.startswith(L_PREFIX)]
    for c in l_cols:
        if df[c].dtype != pl.Int64:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False))
        df = df.with_columns(pl.col(c).fill_null(pl.lit(0, dtype=pl.Int64)))
    return df

def add_cyclic(df: pl.DataFrame) -> pl.DataFrame:
    if "hour" in df.columns:
        df = df.with_columns([
            (2*np.pi*pl.col("hour")/24.0).sin().alias("hour_sin"),
            (2*np.pi*pl.col("hour")/24.0).cos().alias("hour_cos"),
        ])
    if "day_of_week" in df.columns:
        df = df.with_columns([
            (2*np.pi*pl.col("day_of_week")/7.0).sin().alias("dow_sin"),
            (2*np.pi*pl.col("day_of_week")/7.0).cos().alias("dow_cos"),
        ])
    return df

def select_numeric(df: pl.DataFrame, drop_cols=set()) -> list[str]:
    feats = []
    for c, dt in zip(df.columns, df.dtypes):
        if c in drop_cols: continue
        if dt in (pl.Int64, pl.Int32, pl.Float64, pl.Float32): feats.append(c)
    return feats

# ===== F1 helpers =====
def f1_from_counts(tp, fp, fn, eps=1e-9):
    prec = tp / (tp + fp + eps); rec = tp / (tp + fn + eps)
    return 2 * prec * rec / (prec + rec + eps)

def f1_at_threshold(y_true, p, thr):
    y_hat = (p >= thr).astype(np.int32)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    return f1_from_counts(tp, fp, fn), tp, fp, fn

def best_f1(y_true, p, grid=None):
    if grid is None: grid = np.linspace(0.05, 0.95, 91)
    best = (-1.0, 0.5, 0, 0, 0)
    for thr in grid:
        f1, tp, fp, fn = f1_at_threshold(y_true, p, thr)
        if f1 > best[0]: best = (f1, thr, tp, fp, fn)
    return best

# ===== Load AE derived =====
print("[load] AE(train):", TRAIN_AE)
df_tr_ae = pl.read_parquet(TRAIN_AE)      # row_idx 포함
print("[load] AE(test):", TEST_AE)
df_te_ae = pl.read_parquet(TEST_AE)       # ID 포함

# ===== Optionally load tab =====
if USE_TAB:
    print("[load] train_noseq:", TRAIN_NS)
    df_tr = pl.read_parquet(TRAIN_NS).with_row_index("row_idx")
    print("[load] test_noseq:", TEST_NS)
    df_te = pl.read_parquet(TEST_NS)

    # tab preprocess
    df_tr = add_cyclic(cast_l_feats_int(cast_object_int(df_tr)))
    df_te = add_cyclic(cast_l_feats_int(cast_object_int(df_te)))

    # join
    df_tr_full = df_tr.join(df_tr_ae, on="row_idx", how="inner")
    df_te_full = df_te.join(df_te_ae, on="ID", how="inner")
else:
    # AE만 사용: train clicked/row_idx만 붙이고, test는 AE+ID 그대로
    print("[load] train_noseq (labels only):", TRAIN_NS)
    df_lbl = pl.read_parquet(TRAIN_NS, columns=["clicked"]).with_row_index("row_idx")
    df_tr_full = df_lbl.join(df_tr_ae, on="row_idx", how="inner")
    df_te_full = df_te_ae  # ID 포함

print("[shape] train:", df_tr_full.shape, "test:", df_te_full.shape)

# ===== Negative downsampling (train only, when USE_TAB=True to save memory) =====
if USE_TAB and NEG_SAMPLE_FRAC < 1.0:
    y_all = df_tr_full["clicked"].to_numpy()
    pos_mask = (y_all == 1)
    neg_mask = ~pos_mask
    keep_neg = (np.random.rand(neg_mask.sum()) < NEG_SAMPLE_FRAC)
    mask = np.zeros_like(y_all, dtype=bool)
    mask[np.where(pos_mask)[0]] = True
    mask[np.where(neg_mask)[0][keep_neg]] = True
    before = df_tr_full.height
    df_tr_full = df_tr_full.with_row_index("_tmp_idx").filter(pl.col("_tmp_idx").is_in(np.where(mask)[0])).drop("_tmp_idx")
    after = df_tr_full.height
    print(f"[downsample] kept {after}/{before} rows ({after/before:.2%}) "
          f"= pos:{pos_mask.sum()} + neg:{keep_neg.sum()}")

# ===== Feature columns =====
drop_tr = {"row_idx", "clicked"}
drop_te = {"ID"}
feat_cols_tr = select_numeric(df_tr_full, drop_tr)
feat_cols_te = select_numeric(df_te_full, drop_te)
common_feats = sorted(list(set(feat_cols_tr) & set(feat_cols_te)))
print(f"[features] {len(common_feats)} cols; sample={common_feats[:8]}")

# dtype을 미리 float32로 캐스팅(Polars 내부에서) → to_numpy 시 메모리 절감
df_tr_full = df_tr_full.with_columns([pl.col(c).cast(pl.Float32, strict=False) for c in common_feats] + [pl.col("clicked")])
df_te_full = df_te_full.with_columns([pl.col(c).cast(pl.Float32, strict=False) for c in common_feats])

# ===== Numpy conversion (mem-safe) =====
X = df_tr_full.select(common_feats).to_numpy()         # 이미 float32
y = df_tr_full["clicked"].to_numpy().astype(np.int32)
X_test = df_te_full.select(common_feats).to_numpy()    # 이미 float32
test_ids = (df_te_full["ID"].to_numpy() if "ID" in df_te_full.columns else np.arange(X_test.shape[0]))

# ===== Train/valid split =====
n = X.shape[0]; cut = int(n*0.9)
X_tr, y_tr = X[:cut], y[:cut]
X_va, y_va = X[cut:], y[cut:]
print("[split] train:", X_tr.shape, "valid:", X_va.shape)

# 불균형 보정
pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
spw = (neg / max(pos, 1.0))
print(f"[imbalance] scale_pos_weight ≈ {spw:.2f}")

# ===== LightGBM train =====
train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=common_feats, free_raw_data=False)
valid_set = lgb.Dataset(X_va, label=y_va, feature_name=common_feats, free_raw_data=False, reference=train_set)
params = dict(
    objective="binary", metric=["auc","binary_logloss"],
    learning_rate=0.05, num_leaves=64, max_depth=-1,
    min_data_in_leaf=200, feature_fraction=0.9,
    bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0, verbosity=-1,
    boosting_type="gbdt", scale_pos_weight=spw,
)
print("[lgbm] training...")
model = lgb.train(
    params, train_set,
    valid_sets=[train_set, valid_set], valid_names=["train","valid"],
    num_boost_round=5000,
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True),
               lgb.log_evaluation(50)],
)

# ===== Validation F1 =====
p_va = model.predict(X_va, num_iteration=model.best_iteration)
f1_05, tp, fp, fn = f1_at_threshold(y_va, p_va, 0.5)
best_f1_val, best_thr, btp, bfp, bfn = best_f1(y_va, p_va)
print(f"[valid] F1@0.5={f1_05:.4f} (tp={tp}, fp={fp}, fn={fn})")
print(f"[valid] Best-F1={best_f1_val:.4f} at thr={best_thr:.3f} (tp={btp}, fp={bfp}, fn={bfn})")
print("[lgbm] best_iter:", model.best_iteration, "AUC(valid):", model.best_score["valid"]["auc"])

# ===== Save artifacts =====
fi = pl.DataFrame({"feature": common_feats,
                   "gain": model.feature_importance(importance_type="gain"),
                   "split": model.feature_importance(importance_type="split")}).sort("gain", descending=True)
fi.write_csv(FI_CSV); model.save_model(MODEL_TXT)

# ===== Predict test & save submission =====
pred = model.predict(X_test, num_iteration=model.best_iteration)
pl.DataFrame({"ID": test_ids, "clicked": pred}).write_csv(SUB_CSV)
print("[done] saved:", SUB_CSV, "rows=", len(pred))
