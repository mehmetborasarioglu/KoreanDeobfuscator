#!/usr/bin/env python3
"""
evaluate_final_model_g2p.py

Evaluates the G2P-enhanced KoBART model saved in 'final_model_g2p' on 1000 random samples
from the dataset using BLEU, CER, and WER metrics.
"""
import os
import random
import torch
from g2pk import G2p
from evaluate import load as load_metric
from transformers import BartTokenizerFast, BartForConditionalGeneration

MODEL_DIR      = "./checkpoints/final_model_g2p"
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
assert len(orig_lines) == len(obf_lines), "Mismatch in line counts"

indices = random.sample(range(len(orig_lines)), min(NUM_SAMPLES, len(orig_lines)))
sampled_orig = [orig_lines[i] for i in indices]
sampled_obf  = [obf_lines[i]  for i in indices]

g2p = G2p()
tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
model     = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

bleu_metric = load_metric('sacrebleu')
cer_metric  = load_metric('cer')
wer_metric  = load_metric('wer')

all_preds = []
all_refs  = []

for obf_text, orig_text in zip(sampled_obf, sampled_orig):
    phon_input = ' '.join(g2p(obf_text))
    enc = tokenizer(
        phon_input,
        return_tensors='pt',
        truncation=True,
        max_length=None
    ).to(DEVICE)
    input_ids = enc['input_ids']
    attention_mask = enc.get('attention_mask', None)

    generate_kwargs = {
        'max_length': input_ids.shape[1],
        'num_beams': 5,
        'early_stopping': True,
        'no_repeat_ngram_size': 2,
        'length_penalty': 1.0
    }
    if attention_mask is not None:
        generate_kwargs['attention_mask'] = attention_mask

    out_ids = model.generate(input_ids, **generate_kwargs)
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
    print(f"Obfuscated        : {sampled_obf[idx]}")
    print(f"Phonetic Input    : {' '.join(g2p(sampled_obf[idx]))}")
    print(f"Predicted         : {all_preds[idx]}")
    print(f"Original Reference: {all_refs[idx]}")
    print('-' * 40)
