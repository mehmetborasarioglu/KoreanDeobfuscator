#!/usr/bin/env python3
"""
check_pipeline.py

A standalone script to validate each step of the Korean deobfuscation data pipeline:
 1) File loading and line counts
 2) DataFrame integrity
 3) Train/validation split sanity
 4) Dataset conversion checks
 5) Tokenization correctness
 6) Embedding extraction correctness
"""
import os
import sys
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

try:
    from datasets import Dataset
except ImportError:
    print("Error: Install the 'datasets' package (pip install datasets)")
    sys.exit(1)

try:
    from transformers import BartTokenizer, BartForConditionalGeneration
except ImportError:
    print("Error: Install the 'transformers' package (pip install transformers)")
    sys.exit(1)

ORIG_PATH = "data/original.txt"
OBF_PATH  = "data/obfuscated.txt"
MODEL_ID  = "gogamza/kobart-base-v2"
MAX_LEN   = 128
VALID_FRAC = 0.1
SAMPLE_BATCH_SIZE = 2


def main():
    print("[1/6] Checking file existence and line counts...")
    assert os.path.exists(ORIG_PATH), f"Missing file: {ORIG_PATH}"
    assert os.path.exists(OBF_PATH),  f"Missing file: {OBF_PATH}"

    with open(ORIG_PATH, 'r', encoding='utf-8') as f:
        originals = [l.strip() for l in f if l.strip()]
    with open(OBF_PATH, 'r', encoding='utf-8') as f:
        obfuscated = [l.strip() for l in f if l.strip()]

    assert len(originals) == len(obfuscated), (
        f"Line count mismatch: {len(originals)} vs {len(obfuscated)}"
    )
    print(f"    Loaded {len(originals)} lines in both files.")

    print("[2/6] Validating DataFrame integrity...")
    df = pd.DataFrame({'original': originals, 'obfuscated': obfuscated})
    assert not df.isnull().any().any(), "DataFrame contains nulls!"
    assert all(isinstance(x, str) and x for x in df['original']), "Empty or non-str in originals"
    print("    DataFrame has no nulls and all entries are non-empty strings.")

    print("[3/6] Checking train/validation split...")
    train_df, val_df = train_test_split(df, test_size=VALID_FRAC, random_state=42)
    assert len(train_df) + len(val_df) == len(df), "Split size mismatch"
    assert set(train_df.index).isdisjoint(set(val_df.index)), "Train/val indices overlap"
    print(f"    Train size={len(train_df)}, Val size={len(val_df)}")

    print("[4/6] Converting to HuggingFace Dataset and checking...")
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))
    assert train_ds.num_rows == len(train_df), "Dataset train.num_rows mismatch"
    assert val_ds.num_rows == len(val_df),     "Dataset val.num_rows mismatch"
    expected_cols = {'original','obfuscated'}
    assert set(train_ds.column_names) == expected_cols, "Train DS columns wrong"
    assert set(val_ds.column_names)   == expected_cols, "Val DS columns wrong"
    print(f"    Datasets have correct rows and columns: {expected_cols}")

    print("[5/6] Loading tokenizer and validating tokenization...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_ID)
    batch = train_ds.select(range(SAMPLE_BATCH_SIZE))
    enc = tokenizer(
        batch['obfuscated'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )
    ids = enc['input_ids']
    masks = enc['attention_mask']
    assert len(ids) == SAMPLE_BATCH_SIZE, "Batch size mismatch"
    assert all(len(seq)==MAX_LEN for seq in ids), "Tokenized seq length wrong"
    assert all(len(m)==MAX_LEN for m in masks),   "Mask length wrong"
    vocab_size = tokenizer.vocab_size
    flat_ids = [i for seq in ids for i in seq]
    assert all(0 <= token < vocab_size for token in flat_ids), "Token ID out of range"
    decoded = tokenizer.decode(ids[0], skip_special_tokens=True)
    assert isinstance(decoded, str) and decoded, "Decoded text empty"
    print("    Tokenization produces correct shapes, IDs in range, and decodable output.")

    print("[6/6] Loading model and validating embeddings...")
    model = BartForConditionalGeneration.from_pretrained(MODEL_ID)
    input_ids = torch.tensor(ids)
    emb_layer = model.get_input_embeddings()
    embeddings = emb_layer(input_ids)
    b, L, D = embeddings.shape
    assert b == SAMPLE_BATCH_SIZE, "Embedding batch dim mismatch"
    assert L == MAX_LEN, "Embedding seq length mismatch"
    assert D == model.config.d_model, "Embedding hidden dim mismatch"
    assert torch.isfinite(embeddings).all(), "Embeddings contain NaN or Inf"
    assert emb_layer.weight.requires_grad, "Embeddings are frozen!"
    print(f"    Embeddings shape: {embeddings.shape}, all finite, trainable.")

    print("\nAll checks passed! Pipeline is working correctly.")


if __name__ == '__main__':
    main()