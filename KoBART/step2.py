#!/usr/bin/env python3
"""
step2.py

Fine-tune KoBART for Korean deobfuscation with phonetic preprocessing (G2P),
addressing warning messages and optimizing dataset mapping.
"""
import os
import sys
import torch
import pandas as pd
from datasets import Dataset
from g2pk import G2p
from evaluate import load as load_metric
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split

# Configuration
ORIG_PATH   = "data/original.txt"
OBF_PATH    = "data/obfuscated.txt"
MODEL_ID    = "gogamza/kobart-base-v2"
MAX_LEN     = 128
VALID_FRAC  = 0.1
OUTPUT_DIR  = "./checkpoints"
BATCH_SIZE  = 8
NUM_EPOCHS  = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize G2P and tokenizer as globals to avoid closure hash issues
g2p = G2p()
tokenizer = BartTokenizerFast.from_pretrained(MODEL_ID)

# Load configuration and remove any classification metadata to suppress warnings
config = BartConfig.from_pretrained(MODEL_ID)
# Remove classification attributes
if hasattr(config, 'id2label'):
    del config.id2label
if hasattr(config, 'label2id'):
    del config.label2id
# Ensure num_labels is zero
config.num_labels = 0

# Top-level preprocessing function for datasets.map
def preprocess_batch(examples):
    # Convert obfuscated text to phonetic jamo sequences
    phonetic_inputs = [' '.join(g2p(text)) for text in examples['obfuscated']]
    # Tokenize phonetic inputs
    enc = tokenizer(
        phonetic_inputs,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )
    # Tokenize original text for labels
    dec = tokenizer(
        examples['original'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )
    # Mask pad tokens in labels
    labels = [[(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
              for seq in dec['input_ids']]
    enc['labels'] = labels
    return enc

# Compute metrics
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple): preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Restore pad tokens for labels
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in seq] for seq in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Compute BLEU, CER, WER
    bleu = load_metric('sacrebleu').compute(
        predictions=decoded_preds,
        references=[[ref] for ref in decoded_labels]
    )['score']
    cer = load_metric('cer').compute(predictions=decoded_preds, references=decoded_labels)
    wer = load_metric('wer').compute(predictions=decoded_preds, references=decoded_labels)
    return {'bleu': bleu, 'cer': cer, 'wer': wer}

if __name__ == '__main__':
    # Load raw data
    originals = [l.strip() for l in open(ORIG_PATH, encoding='utf-8') if l.strip()]
    obfuscated = [l.strip() for l in open(OBF_PATH, encoding='utf-8') if l.strip()]
    df = pd.DataFrame({'original': originals, 'obfuscated': obfuscated})
    train_df, val_df = train_test_split(df, test_size=VALID_FRAC, random_state=42)

    # Create HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

    # Tokenize with phonetic preprocessing
    train_tok = train_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=train_ds.column_names,
        num_proc=4
    )
    val_tok = val_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=val_ds.column_names,
        num_proc=4
    )

    # Load model with cleaned config
    model = BartForConditionalGeneration.from_pretrained(
        MODEL_ID,
        config=config
    ).to(DEVICE)

    # Set up training
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE*2,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        predict_with_generate=True,
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='bleu',
        greater_is_better=True
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    print(metrics)

    # Sample predictions
    print("\nSample predictions with phonetic input:")
    for idx in range(min(5, len(val_ds))):
        ex = val_ds[idx]
        phon_input = ' '.join(g2p(ex['obfuscated']))
        inputs = tokenizer(
            phon_input,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN
        ).to(DEVICE)
        out = model.generate(
            **inputs,
            max_length=MAX_LEN,
            num_beams=5,
            no_repeat_ngram_size=2
        )
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"Example {idx+1}:")
        print(f"Obfuscated phonetic: {phon_input}")
        print(f"Predicted          : {pred}")
        print(f"Original           : {ex['original']}")
        print("-"*40)

    # Save
    trainer.save_model(os.path.join(OUTPUT_DIR, 'final_model_g2p'))
    print("Done.")
