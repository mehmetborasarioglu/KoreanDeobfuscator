import fs from 'fs';
import readline from 'readline';
import { uglifyText } from './hangulObfuscator.js'; 

const AS_IT_SOUNDS_DEGREE = 50;
const DOUBLE_CONSONANT_AS_END_SOUND_DEGREE = 50;
const COMPONENT_CHANGE_DEGREE = 50;
const ADD_EXTRA_END_SOUND_DEGREE = 50;
const ONLY_WANSUNG_CHARS = true;

/**
 * Reads up to 1000 lines from an input text file (UTF-8),
 * obfuscates each line, and writes out a CSV with two columns:
 * "original","obfuscated".
 */
async function createParallelDatasetShort(inputFile, outputFile) {
  const readStream = fs.createReadStream(inputFile, { encoding: 'utf-8' });
  const rl = readline.createInterface({ input: readStream });
  const writeStream = fs.createWriteStream(outputFile, { encoding: 'utf-8' });

  writeStream.write('original,obfuscated\n');

  let count = 0; 

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue; 

    const obfuscated = uglifyText(
      trimmed,
      AS_IT_SOUNDS_DEGREE,
      DOUBLE_CONSONANT_AS_END_SOUND_DEGREE,
      COMPONENT_CHANGE_DEGREE,
      ADD_EXTRA_END_SOUND_DEGREE,
      ONLY_WANSUNG_CHARS
    );

    const originalEscaped = trimmed.replace(/"/g, '""');
    const obfEscaped = obfuscated.replace(/"/g, '""');

    writeStream.write(`"${originalEscaped}","${obfEscaped}"\n`);

    count += 1;
    if (count >= 1000) {
      break;
    }
  }

  writeStream.end();
}

const INPUT_FILE = 'input.txt';
const OUTPUT_FILE = 'parallel_dataset_limited.csv';

createParallelDatasetShort(INPUT_FILE, OUTPUT_FILE)
  .then(() => {
    console.log(`Done. Processed up to 1000 lines. Check ${OUTPUT_FILE}.`);
  })
  .catch(err => {
    console.error('Error:', err);
  });
