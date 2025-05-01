import fs from 'fs';
import readline from 'readline';
import { uglifyText } from './hangulObfuscator.js'; 

const AS_IT_SOUNDS_DEGREE = 50;
const DOUBLE_CONSONANT_AS_END_SOUND_DEGREE = 50;
const COMPONENT_CHANGE_DEGREE = 50;
const ADD_EXTRA_END_SOUND_DEGREE = 50;
const ONLY_WANSUNG_CHARS = true;

const MAX_LINES = 10_000_000;  

async function obfuscateToTextWithLimit(inputFile, outputFile) {
  const readStream = fs.createReadStream(inputFile, { encoding: 'utf-8' });
  const rl = readline.createInterface({ input: readStream });
  const writeStream = fs.createWriteStream(outputFile, { encoding: 'utf-8' });

  let lineCount = 0;

  for await (const line of rl) {
    if (lineCount >= MAX_LINES) {
      console.log(`Reached ${MAX_LINES} lines. Stopping early.`);
      break;
    }

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

    writeStream.write(obfuscated + '\n');
    lineCount += 1;
  }

  writeStream.end();
  console.log(`Done. Processed ${lineCount} lines (max ${MAX_LINES}). Output: ${outputFile}`);
}


const INPUT_FILE = 'input.txt';   
const OUTPUT_FILE = 'output.txt'; 

obfuscateToTextWithLimit(INPUT_FILE, OUTPUT_FILE).catch(err => {
  console.error('Error:', err);
});
