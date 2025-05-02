# KoBART Fine-Tuning & Evaluation Pipeline

This directory contains scripts for preprocessing, training, and evaluating a Korean deobfuscation model using the pre-trained KoBART (`gogamza/kobart-base-v2`). The process integrates phonetic preprocessing (G2P) and computes standard text similarity metrics.

## Files

- `step1.py`: Validates the data pipeline (loading, tokenization, embedding sanity checks)
- `step2.py`: Fine-tunes the KoBART model on phonetic-preprocessed input
- `evaluation.py`: Evaluates a trained model on a 1000-sample subset and computes BLEU, CER, and WER

## Requirements

Install dependencies using:

```bash
pip install -r requirements_kobart.txt
```

## Data Format

These scripts expect:

- `data/original.txt`: Standard Korean text
- `data/obfuscated.txt`: Matching obfuscated text (same number of lines)

Each line in both files should correspond to a single example.

## Usage

### 1. Run pipeline checks
```bash
python step1.py
```

### 2. Fine-tune KoBART with phonetic preprocessing
```bash
python step2.py
```

This script uses G2P conversion (via `g2pk`) and fine-tunes KoBART using HuggingFace's `Seq2SeqTrainer`.

### 3. Evaluate model performance
```bash
python evaluation.py
```

This runs predictions on a 1000-line subset and outputs BLEU, CER, and WER scores.

---

These scripts support the research project:  
**"Korean Airbnb Script Deobfuscation Using Sequence-to-Sequence Models"**  
by Abdul Rafay & Mehmet Bora Sarioglu (Boston University)
