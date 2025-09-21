# make_seq_vocab.py
import pandas as pd
import json
from typing import List, Dict

PAD, UNK = 0, 1
VOCAB_JSON = "./seq_vocab_full.json"

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")

def build_full_vocab(seqs: List[List[int]]) -> Dict[int, int]:
    uniq = set()
    for s in seqs:
        uniq.update(s)
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    next_id = 2
    for tok in sorted(uniq):
        vocab[int(tok)] = next_id
        next_id += 1
    return vocab

def save_vocab(vocab: Dict[int, int], path: str):
    dump = {str(k): v for k, v in vocab.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False)

if __name__ == "__main__":
    # train_seq.parquet만 읽으면 됨 (train에 있는 unique 토큰 전부 반영)
    TRAIN_SEQ = "../data/train_seqlist_tot.parquet"

    print("▶ train_seq 읽는 중...")
    tr_seq = read_parquet(TRAIN_SEQ)
    assert "seq" in tr_seq.columns, "seq column missing"

    train_seqs: List[List[int]] = tr_seq["seq"].tolist()
    print(f"총 샘플 수: {len(train_seqs)}")

    # vocab 생성
    print("▶ vocab 생성 중...")
    vocab = build_full_vocab(train_seqs)
    print(f"Unique 토큰 수: {len(vocab) - 2} (PAD/UNK 제외)")

    # 저장
    save_vocab(vocab, VOCAB_JSON)
    print(f"✅ vocab 저장 완료: {VOCAB_JSON}")
