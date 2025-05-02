# KoBART-G2P Training & Evaluation Pipeline

This directory contains scripts and configuration for validating, training, and evaluating the G2P-enhanced KoBART deobfuscation model.

## Files

- **requirements.txt**  
  Python dependencies needed for all scripts:
  ```
  torch>=1.12.0
  transformers>=4.36.0
  datasets>=2.18.0
  evaluate>=0.4.0
  g2pk>=0.9.4
  scikit-learn>=1.4.0
  pandas>=2.0.0
  ```

- **check_pipeline.py.py**  
  Standalone sanity checks for the data pipeline. Validates:
  1. Existence and line counts of `data/original.txt` & `data/obfuscated.txt`  
  2. DataFrame integrity (no nulls, correct types)  
  3. Train/validation split correctness  
  4. Conversion to HuggingFace `datasets.Dataset`  
  5. Tokenization shapes, ID ranges, and decode round-trip  
  6. Embedding extraction for BART embeddings  
  ```bash
  python check_pipeline.py.py
  ```

- **train_model.py**  
  Fine‑tunes `gogamza/kobart-base-v2` with G2P phonetic preprocessing:
  - Converts obfuscated text → phonetic jamo via `g2pk`
  - Tokenizes inputs & targets with `BartTokenizerFast`
  - Trains a Seq2Seq model using `Seq2SeqTrainer`
  - Saves best model to `./checkpoints/final_model_g2p`
  ```bash
  python train_model.py
  ```

- **evalutation_g2p.py**  
  Evaluates the saved G2P model on a random subset:
  - Loads `final_model_g2p` checkpoints
  - Measures BLEU, CER, WER on N=100 000 samples
  - Prints summary metrics & a few random examples  
  ```bash
  python evalutation_g2p.py
  ```

##  Usage

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Validate pipeline**  
   ```bash
   python check_pipeline.py.py
   ```

3. **Train the model**  
   ```bash
   python train_model.py
   ```

4. **Evaluate performance**  
   ```bash
   python evalutation_g2p.py
   ```

---

Part of “Korean Airbnb Script Deobfuscation Using Sequence-to-Sequence Models.”  
