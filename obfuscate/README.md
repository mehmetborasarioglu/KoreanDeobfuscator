# Hangul Obfuscator Scripts

This directory contains JavaScript scripts to generate obfuscated versions of Korean text using a variety of phonetic and sub-character transformations. The obfuscation mimics common stylistic distortions seen in online Korean platforms.

## Installation

Install dependencies with:

```bash
npm install
```

## Files and Purpose

- `hangulObfuscator.js`: Core module exporting `uglifyText`, which performs complex Jamo-level transformations
- `obfuscate.js`: Example usage of `uglifyText` on a test sentence
- `copyFirst10M.js`: Extracts the first 10 million lines from a large input file
- `createParallelDataset.js`: Obfuscates up to 10M lines using `uglifyText` and writes parallel text (input/output) files
- `createParallelDatasetShort.js`: Generates a small parallel CSV with 1000 lines for quick inspection

## Parameters for Obfuscation

The `uglifyText()` function accepts:

- `asItSoundsDegree`: Probability of mimicking real phonetic patterns
- `doubleConsonantAsEndSoundDegree`: Chance of doubling consonants in end syllables
- `componentChangeDegree`: Likelihood of altering individual Jamo characters
- `addExtraEndSoundDegree`: Chance of adding final consonants to syllables that lack them
- `onlyWansungChars`: Whether to restrict output to complete Hangul syllables only

## Example

```bash
node obfuscate.js
```

You’ll see:
```
Original: 안녕하세요! 한글 텍스트 난독화를 시도해 봅시다.
Obfuscated: 앀냗핮먃! 핱긇 텋슷 늣퍄핟를 씰도흫 봅쉎닿.
```

## Dataset Generation Workflow

```bash
node copyFirst10M.js              # Extracts original.txt from raw input
node createParallelDataset.js     # Creates output.txt (obfuscated text)
```

Optional for small test:
```bash
node createParallelDatasetShort.js
# Creates parallel_dataset_limited.csv
```

---

Part of the preprocessing pipeline for the research project:  
**"Korean Airbnb Script Deobfuscation Using Sequence-to-Sequence Models"**  
by Abdul Rafay & Mehmet Bora Sarioglu (Boston University)
