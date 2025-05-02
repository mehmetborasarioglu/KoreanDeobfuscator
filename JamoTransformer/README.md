# Hierarchical Jamo Transformer (HJST)

This directory contains the implementation of a custom Hierarchical Transformer architecture designed to decode Korean "Airbnb script" at the sub-character level, utilizing both **Jamo-level** and **syllable-level** linguistic structures.

## Contents

- `hierarchical_jamo_transformer.py`: Full transformer model with custom cross-level attention and generation logic
- `tokenizer.py`: JamoTokenizer that decomposes Hangul syllables into character components (초성/중성/종성)
- `dataset.py`: PyTorch dataset class for paired Korean obfuscated/original text using the custom tokenizer

## Features

- **Hierarchical Encoding**: Models relationships across jamo and syllable structures
- **Position-Aware Decoder**: Predicts and adjusts for jamo position (initial/medial/final)
- **Sub-character Tokenization**: Maintains fidelity during decoding using `jamo` module
- **Custom Loss Function**: Blends cross-entropy and position loss for Korean-aware decoding
- **BLEU, CER, WER Evaluation**

## Usage

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python hierarchical_jamo_transformer.py
```

This will:
- Build the tokenizer from both obfuscated and original text
- Train the model using a 90/10 train/val split
- Evaluate the model and print BLEU, CER, WER metrics

### 3. Data Format

Requires two files:
- `data/obfuscated_small.txt`
- `data/original_small.txt`

These should be line-aligned (same number of lines), and contain Korean text.

### 4. Output

- Checkpoints are saved to `checkpoints/`
- Best model: `checkpoints/hjst_best.pth`

---

This module implements the experimental architecture described in:

**"Korean Airbnb Script Deobfuscation Using Sequence-to-Sequence Models"**  
by Abdul Rafay & Mehmet Bora Sarioglu (Boston University)
