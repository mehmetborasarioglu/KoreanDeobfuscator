export class KoreanTextDecoder {
  async deobfuscate(text) {
    const res = await fetch('/api/decode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error(await res.text());
    const { decoded } = await res.json();
    return decoded;
  }

  
  async translate(text) {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error(await res.text());
    const { translation } = await res.json();
    return translation;
  }
}
