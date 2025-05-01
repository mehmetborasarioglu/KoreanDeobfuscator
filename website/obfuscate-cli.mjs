#!/usr/bin/env node
import { uglifyText } from './static/obfuscate.js';

const text = process.argv[2] || '';
const obf = uglifyText(
  text,
  50,  
  50,  
  70,  
  30,  
  true 
);

console.log(obf);
