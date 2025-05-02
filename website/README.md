# Korean Deobfuscation Web App

This directory contains the full backend server and frontend integration for the **Korean Airbnb Script Deobfuscation** project. It allows users to submit obfuscated Korean text, receive deobfuscated and translated output, and also test obfuscation transformations via JavaScript.

## 🚀 Features

- **/api/obfuscate**: Applies Hangul-level transformations (Airbnb-style obfuscation)
- **/api/decode**: Decodes obfuscated Korean into standard Korean using a fine-tuned BART model
- **/api/translate**: Translates Korean into English using MarianMT

## 🛠 Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
npm install
```

## 📦 Download Pretrained Model (Required)

The deobfuscation model (`./final_model/`) is too large to include in the GitHub repo.

📥 Please download it manually from:

👉 **[Google Drive Model Link](https://drive.google.com/drive/folders/1RznwsSYVs0_l8TLJpN0eBvz4MLjnrLo1?usp=drive_link)**

Extract the folder and place it inside this directory as:

```
./final_model/
├── config.json
├── tokenizer.json
├── pytorch_model.bin
...
```

## ▶️ Run the Server

```bash
uvicorn server:app --reload
```

Visit: [http://localhost:8000](http://localhost:8000)

## 🧪 CLI Test (Optional)

```bash
node obfuscate-cli.mjs "텍스트를 입력하세요"
```

## Folder Structure

- `server.py` – FastAPI app serving 3 endpoints
- `hangulObfuscator.js` – JavaScript module to simulate Korean obfuscation
- `obfuscate-cli.mjs` – CLI demo for obfuscation
- `static/` – (Optional) frontend HTML/JS files
- `final_model/` – pretrained BART-based deobfuscation model (see above)

---

Part of the final project:  
**“Korean Airbnb Script Deobfuscation Using Sequence-to-Sequence Models”**  
by Abdul Rafay & Mehmet Bora Sarioglu (Boston University)
