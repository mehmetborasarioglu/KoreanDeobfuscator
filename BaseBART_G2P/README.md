# KoreanDeobfuscator – Model Training Module

This directory contains the main training script for the Korean Airbnb Script Deobfuscation project. The `model.py` script is responsible for building, training, and evaluating a character-level sequence-to-sequence model (based on BART) to decode phonetic Korean text obfuscated using "Airbnb-style" transformations.

## Overview

The training pipeline includes:
- **Phonetic Preprocessing** using `g2pK`
- **Custom Character-Level Tokenizer** with special symbols (`<pad>`, `<unk>`, `<s>`, `</s>`)
- **Dataset Preparation** from aligned obfuscated-original pairs
- **BART Model Configuration** with 6 encoder and decoder layers
- **Training with HuggingFace Transformers** using `Seq2SeqTrainer`
- **Metric Tracking** (accuracy based on exact token match)

## Directory Structure

- `data/original.txt`: Ground-truth standard Korean text
- `data/obfuscated.txt`: Corresponding obfuscated Korean text
- `cache/`: Automatically generated cache files for tokenizers and datasets
- `checkpoints/`: Directory where trained model weights are saved
- `logs/`: Directory for training logs

## Installation

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Script

To train the model from scratch:

```bash
python model.py
```

You can configure the maximum number of training samples via the `max_samples` parameter inside the script.

## Notes

- The phonetic preprocessing step uses `g2pK` to convert both original and obfuscated text into more comparable phoneme-like forms.
- The model is trained using character-level tokenization, allowing it to generalize better over unseen obfuscation patterns.
- Training progress and evaluation accuracy are logged live using `tqdm`-enhanced callbacks.

## Output

After training:
- The model is saved in `checkpoints/final_model/`
- Sample deobfuscations are printed for 10 randomly selected validation entries.

---

This script is a core component of the research presented in the paper:  
**“Korean Airbnb Script Deobfuscation Using Sequence-to-Sequence Models”**  
by Abdul Rafay & Mehmet Bora Sarioglu (Boston University)
