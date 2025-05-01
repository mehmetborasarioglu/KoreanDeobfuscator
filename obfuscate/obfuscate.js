// File: obfuscate.js

// 1) Import the 'uglifyText' function from your original source file:
import { uglifyText } from './hangulObfuscator.js';

/**
 * Example usage / demonstration of calling the uglifyText function
 * with parameter values for:
 * - asItSoundsDegree
 * - doubleConsonantAsEndSoundDegree
 * - componentChangeDegree
 * - addExtraEndSoundDegree
 * - onlyWansungChars
 */
function main() {
  // 2) The text you want to obfuscate:
  const textToObfuscate = '안녕하세요! 한글 텍스트 난독화를 시도해 봅시다.';

  // 3) Example parameter values:
  const asItSoundsDegree = 50;                 // 0 ~ 100
  const doubleConsonantAsEndSoundDegree = 50;  // 0 ~ 100
  const componentChangeDegree = 70;            // 0 ~ 100
  const addExtraEndSoundDegree = 30;           // 0 ~ 100
  const onlyWansungChars = true;               // true or false

  // 4) Obfuscate it:
  const obfuscated = uglifyText(
    textToObfuscate,
    asItSoundsDegree,
    doubleConsonantAsEndSoundDegree,
    componentChangeDegree,
    addExtraEndSoundDegree,
    onlyWansungChars
  );

  // 5) Show result:
  console.log('Original:', textToObfuscate);
  console.log('Obfuscated:', obfuscated);
}

// 6) Invoke main (or adapt if you prefer a CLI approach)
main();
