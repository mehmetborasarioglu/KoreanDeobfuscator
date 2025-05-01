import { KoreanTextDecoder } from './decode.js';

document.addEventListener('DOMContentLoaded', () => {
  const themeToggle = document.getElementById('themeToggle');
  themeToggle.addEventListener('click', () => {
    const current = document.body.getAttribute('data-theme') || 'light';
    const next    = current === 'light' ? 'dark' : 'light';
    document.body.setAttribute('data-theme', next);
    themeToggle.textContent = next === 'dark' ? '☀️' : '🌙';
  });

  const inputText         = document.getElementById('inputText');
  const obfuscateButton   = document.getElementById('obfuscateButton');
  const obfuscatedText    = document.getElementById('obfuscatedText');
  const deobfuscateButton = document.getElementById('deobfuscateButton');
  const outputText        = document.getElementById('outputText');
  const translateButton   = document.getElementById('translateButton');
  const translationText   = document.getElementById('translationText');

  obfuscateButton.addEventListener('click', async () => {
    const text = inputText.value.trim();
    if (!text) return;

    obfuscateButton.disabled    = true;
    obfuscateButton.textContent = 'Obfuscating…';

    try {
      const res = await fetch('/api/obfuscate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (!res.ok) throw new Error(await res.text());
      const { obfuscated } = await res.json();
      obfuscatedText.value  = obfuscated;
      outputText.value      = '';
      translationText.value = '';
    } catch (err) {
      console.error(err);
      obfuscatedText.value = 'Error obfuscating';
    } finally {
      obfuscateButton.disabled    = false;
      obfuscateButton.textContent = 'Obfuscate';
    }
  });

  deobfuscateButton.addEventListener('click', async () => {
    const text = obfuscatedText.value.trim();
    if (!text) return;

    deobfuscateButton.disabled    = true;
    deobfuscateButton.textContent = 'De-obfuscating…';

    try {
      const decoder = new KoreanTextDecoder();
      const decoded = await decoder.deobfuscate(text);
      outputText.value      = decoded;
      translationText.value = '';
    } catch (err) {
      console.error(err);
      outputText.value = 'Error de-obfuscating';
    } finally {
      deobfuscateButton.disabled    = false;
      deobfuscateButton.textContent = 'De-obfuscate';
    }
  });

  translateButton.addEventListener('click', async () => {
    const text = outputText.value.trim();
    if (!text) return;

    translateButton.disabled    = true;
    translateButton.textContent = 'Translating…';

    try {
      const decoder     = new KoreanTextDecoder();
      const translation = await decoder.translate(text);
      translationText.value = translation;
    } catch (err) {
      console.error(err);
      translationText.value = 'Error translating';
    } finally {
      translateButton.disabled    = false;
      translateButton.textContent = 'Translate';
    }
  });
});
