# Korean “Airbnb Script” Deobfuscator

_A sequence-to-sequence system for reversing the phonetic “Airbnb script” obfuscation used in Korean online texts._

## Repository Structure

```
├── BaseBART_G2P
├── KoBART
├── KoBART_G2P
├── JamoTransformer
├── data
├── obfuscate
├── website
├── milestone2
├── Korean Deobfuscation Poll.csv
└── README.md
```

### `BaseBART_G2P/`  
A from-scratch BART pipeline that:  
- Converts raw & obfuscated Korean to phonetic sequences via `g2pK`.  
- Trains a custom-vocab BART model (see `model.py`).  
- Includes data loaders, tokenizers, training & evaluation scripts.  

### `KoBART/`  
Baseline fine-tuning of `gogamza/kobart-base-v2` on our parallel original↔obfuscated dataset (no phonetic preprocessing).  

### `KoBART_G2P/`  
Improved KoBART pipeline with G2P phonetic conversion in preprocessing (`step2.py`, `evalutation_g2p.py`, etc.).  

### `JamoTransformer/`  
Our hierarchical Jamo–syllable Transformer:  
- `tokenizer.py` – decomposes Hangul → Jamo  
- `dataset.py` – PyTorch Dataset  
- `hierarchical_jamo_transformer.py` – custom encoder/decoder  
- Training & evaluation scripts  

### `data/`  
All of our text data:  
- `original.txt` & `obfuscated.txt` (full / sampled)  
- Generated parallel CSVs, cleaned subsets, etc.  

### `obfuscate/`  
Node.js scripts for generating obfuscated data from raw Korean:  
- `hangulObfuscator.js` – core obfuscation logic  
- `createParallelDataset.js` / `createParallelDatasetShort.js` / `copyFirst10M.js`  
- CLI wrapper `obfuscate-cli.mjs`  

### `website/`  
Full demo app (FastAPI + static frontend):  
- **Backend**: `server.py` hosts `/api/obfuscate`, `/api/decode`, `/api/translate`.  
- **Frontend**: `static/index.html`, `app.js`, `styles.css`.  
- **Note:** The final de-obfuscation model is too large for GitHub.  
  1. Download the `final_model` directory from Google Drive:  
     https://drive.google.com/drive/folders/1RznwsSYVs0_l8TLJpN0eBvz4MLjnrLo1?usp=drive_link  
  2. Place it here:  
     ```
     website/final_model/
     ```  
  3. Install Python deps and run:  
     ```bash
     cd website
     pip install -r requirements.txt
     uvicorn server:app --reload
     ```  

### `milestone2/`  
Deliverables, slides, and notes from our second course milestone.  

### `Korean Deobfuscation Poll.csv`  
Native-speaker survey results on model quality (average rating ≈ 8/10).  

---



## Poll

Check out `Korean Deobfuscation Poll.csv` for native Korean speaker feedback.  

---

## Citation

If you use this work, please cite:  
> Rafay, A. & Sarioglu, M. (2025). _Korean “Airbnb” Script Deobfuscation Using Sequence-to-Sequence Models._ Boston University.

---

## License

MIT © 2025 Abdul Rafay & Mehmet Bora Sarioglu  
