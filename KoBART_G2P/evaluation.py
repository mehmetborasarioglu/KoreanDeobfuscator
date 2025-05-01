#!/usr/bin/env python3
"""
evaluate_random_samples.py

Loads the fine-tuned KoBART model and evaluates it on a random subset of 1000 examples
from the dataset files 'data/original.txt' and 'data/obfuscated.txt'.
Computes BLEU, CER, and WER on those 1000 samples, then prints 10 random examples
showing the obfuscated input, model prediction, and original reference.
"""
import os
import random
import torch
from evaluate import load as load_metric
from transformers import BartTokenizerFast, BartForConditionalGeneration

MODEL_DIR      = "./checkpoints/final_model"
ORIG_FILE      = "data/original.txt"
OBF_FILE       = "data/obfuscated.txt"
NUM_SAMPLES    = 100000
NUM_SHOW       = 10
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)

with open(ORIG_FILE, encoding='utf-8') as f:
    orig_lines = [l.strip() for l in f if l.strip()]
with open(OBF_FILE, encoding='utf-8') as f:
    obf_lines = [l.strip() for l in f if l.strip()]
assert len(orig_lines) == len(obf_lines), "Original and obfuscated files must have the same number of lines"

indices = random.sample(range(len(orig_lines)), min(NUM_SAMPLES, len(orig_lines)))
sampled_orig = [orig_lines[i] for i in indices]
sampled_obf  = [obf_lines[i]  for i in indices]

tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
model     = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

bleu_metric = load_metric('sacrebleu')
cer_metric  = load_metric('cer')
wer_metric  = load_metric('wer')

all_preds = []
all_refs  = []

for obf_text, orig_text in zip(sampled_obf, sampled_orig):
    enc = tokenizer(
        obf_text,
        return_tensors='pt',
        truncation=True,
        max_length=None
    ).to(DEVICE)
    input_ids = enc['input_ids']
    attention_mask = enc.get('attention_mask', None)
    seq_len = input_ids.shape[1]

    generate_kwargs = {
        'max_length': seq_len,
        'num_beams': 5,
        'early_stopping': True,
        'no_repeat_ngram_size': 2,
        'length_penalty': 1.0
    }
    if attention_mask is not None:
        generate_kwargs['attention_mask'] = attention_mask

    out_ids = model.generate(
        input_ids,
        **generate_kwargs
    )
    pred_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    max_chars = len(obf_text)
    if len(pred_text) > max_chars:
        pred_text = pred_text[:max_chars]

    all_preds.append(pred_text)
    all_refs.append(orig_text)

bleu = bleu_metric.compute(predictions=all_preds, references=[[r] for r in all_refs])['score']
cer  = cer_metric.compute(predictions=all_preds, references=all_refs)
wer  = wer_metric.compute(predictions=all_preds, references=all_refs)

print(f"Evaluated on {len(all_refs)} random samples:")
print(f"BLEU: {bleu:.2f}")
print(f"CER : {cer:.4f}")
print(f"WER : {wer:.4f}")

print(f"\nShowing {NUM_SHOW} random examples:\n")
show_idxs = random.sample(range(len(all_preds)), min(NUM_SHOW, len(all_preds)))
for idx in show_idxs:
    print(f"Example {idx+1}:")
    print(f"Obfuscated: {sampled_obf[idx]}")
    print(f"Predicted : {all_preds[idx]}")
    print(f"Original  : {all_refs[idx]}")
    print('-' * 40)
