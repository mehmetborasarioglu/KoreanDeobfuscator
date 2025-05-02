##  Downloading and Preparing Training Data

As described in Section 4.1 of our research paper, the training corpus is built from the KoWiki dataset (26.8 M lines, ~1.7 GB) and its obfuscated counterpart. Because of its size, we provide scripts to reproduce and sample it.

### 1. Download the KoWiki dump  
```bash
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 -P data/
```

### 2. Extract plain text  
```bash
git clone https://github.com/attardi/wikiextractor.git
python3 wikiextractor/WikiExtractor.py --no-templates --min_text_length 200 --output data/kowiki-text data/kowiki-latest-pages-articles.xml.bz2
```

This will create `data/kowiki-text/AA/wiki_00`… text files.

### 3. Generate aligned original & obfuscated files  
```bash
# (Optional) Copy only the first 10 million lines for faster prototyping
node obfuscate/copyFirst10M.js data/kowiki-text/all.txt data/original.txt

# Use our obfuscator to create the obfuscated counterpart
node obfuscate/createParallelDataset.js data/original.txt data/obfuscated.txt
```

Result:
- `data/original.txt` — raw Korean sentences  
- `data/obfuscated.txt` — “Airbnb script” versions, line‑aligned

### 4. Create a small sample (1 K examples)  
```bash
node obfuscate/createParallelDatasetShort.js data/original.txt data/parallel_sample.csv
```

Produces `parallel_sample.csv` with columns `original,obfuscated`.

---
