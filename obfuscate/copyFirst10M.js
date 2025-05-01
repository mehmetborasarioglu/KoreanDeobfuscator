import fs from 'fs';
import readline from 'readline';

const MAX_LINES = 10_000_000; 

async function copyFirst10M(inputFile, outputFile) {
  const readStream = fs.createReadStream(inputFile, { encoding: 'utf-8' });
  const rl = readline.createInterface({ input: readStream });
  const writeStream = fs.createWriteStream(outputFile, { encoding: 'utf-8' });

  let lineCount = 0;

  for await (const line of rl) {
    if (lineCount >= MAX_LINES) {
      console.log(`Reached ${MAX_LINES} lines. Stopping early.`);
      break;
    }

    writeStream.write(line + '\n');
    lineCount++;
  }

  writeStream.end();
  console.log(`Done. Copied ${lineCount} lines (max ${MAX_LINES}).`);
}


const INPUT_FILE = 'input.txt';    
const OUTPUT_FILE = 'original.txt';

copyFirst10M(INPUT_FILE, OUTPUT_FILE).catch((err) => {
  console.error('Error:', err);
});
