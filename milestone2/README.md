# Data Processing & Client-Side Decoder

This directory contains scripts and assets for data acquisition, dataset generation, cleaning, and a prototype client-side decoder for the Korean Airbnb Script Deobfuscation project.

## 🚀 Overview

1. **Data Acquisition**  
   - `download.py`: Downloads raw KoWiki and NamuWiki dumps into `data/` using defined URLs.
2. **Dataset Generation**  
   - `data_gen.py`: Uses Selenium to feed raw Korean text to the Hanmesoft Airbnbfy obfuscator, creating `encoded_korean_texts.csv`.
3. **Data Cleaning**  
   - `clean_csv.py`: Reads `encoded_korean_texts.csv`, retains the first two columns (`original`, `obfuscated`), drops nulls, and outputs `clean_encoded_korean_texts.csv`.
4. **Model Prototype Training**  
   - `train.py`: Python training script that loads `clean_encoded_korean_texts.csv`, preprocesses text into fixed-length sequences, and trains a Seq2Seq model (`decoder.py` components) with BLEU evaluation.  
5. **Client-Side Decoder**  
   - `decoder.py`: Python implementation of a Seq2Seq encoder–decoder architecture (with Jamo decomposition via `jamotools`) for experimentation and extension.  
   - `decode.js`: Browser-based `KoreanTextDecoder` stub that can load model weights/client logic.  
   - `app.js`: Frontend integration, wiring UI elements to the decoder API.  
   - `styles.css`: CSS styling for the frontend demo, including theming support and layout.
6. **Example Data**  
   - `clean_encoded_korean_texts.csv`: Sample cleaned CSV of original and obfuscated lines for training.

## 📦 Requirements

Install Python dependencies:

```bash
pip install -r requirements_data_processing.txt
```

Make sure **ChromeDriver** is installed and on your `PATH` for `data_gen.py`.

## 🔗 Model Download

The pretrained deobfuscation model is too large for GitHub. Download it from:

👉 [Google Drive Model Folder](https://drive.google.com/drive/folders/1RznwsSYVs0_l8TLJpN0eBvz4MLjnrLo1?usp=drive_link)

Extract into this directory as `model/` so that `train.py`, `decoder.py`, and client-side code can load it.

## 📋 Usage

1. **Download raw dumps**  
   ```bash
   python download.py
   ```
2. **Generate encoded dataset**  
   ```bash
   python data_gen.py
   ```
3. **Clean dataset**  
   ```bash
   python clean_csv.py
   ```
4. **Train prototype model**  
   ```bash
   python train.py
   ```
5. **Test client-side decoding**  
   - Serve `app.js`, `decode.js`, and `styles.css` in a static folder and open the HTML demo in a browser.

---

This module underpins the data pipeline and in-browser decoding features of the research project:  
**“Korean Airbnb Script Deobfuscation Using Sequence-to-Sequence Models”**  
by Abdul Rafay & Mehmet Bora Sarioglu (Boston University)
