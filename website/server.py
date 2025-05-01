# server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import (
    BartTokenizerFast, BartForConditionalGeneration,
    MarianTokenizer, MarianMTModel,
)
import execjs
import uvicorn


with open("hangulObfuscator.js", "r", encoding="utf-8") as f:
    js_src = f.read()
ctx = execjs.compile(js_src)

DEOBF_MODEL_DIR = "./final_model"
deobf_tokenizer = BartTokenizerFast.from_pretrained(DEOBF_MODEL_DIR)
deobf_model     = BartForConditionalGeneration.from_pretrained(DEOBF_MODEL_DIR)

trans_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
trans_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/node_modules", StaticFiles(directory="node_modules"), name="node_modules")

@app.get("/", include_in_schema=False)
async def homepage():
    return FileResponse("static/index.html")

class ObfuscateRequest(BaseModel):
    text: str

class ObfuscateResponse(BaseModel):
    obfuscated: str

@app.post("/api/obfuscate", response_model=ObfuscateResponse)
async def obfuscate(req: ObfuscateRequest):
    try:
        obf = ctx.call(
            "uglifyText",
            req.text,
            50,    
            50,    
            50,    
            50,    
            True   
        )
        return {"obfuscated": obf}
    except execjs.ProgramError as e:
        raise HTTPException(500, detail=f"Obfuscation JS error: {e}")


class DecodeRequest(BaseModel):
    text: str

class DecodeResponse(BaseModel):
    decoded: str

@app.post("/api/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest):
    enc = deobf_tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    out_ids = deobf_model.generate(
        **enc,
        max_length=enc["input_ids"].shape[1],
        num_beams=5
    )
    full_decoded = deobf_tokenizer.decode(out_ids[0], skip_special_tokens=True)

    
    max_chars = len(req.text)
    truncated = full_decoded[:max_chars]

    return {"decoded": truncated}


class TranslateRequest(BaseModel):
    text: str

class TranslateResponse(BaseModel):
    translation: str

@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    tenc = trans_tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    to_ids = trans_model.generate(**tenc)
    translation = trans_tokenizer.decode(to_ids[0], skip_special_tokens=True)
    return {"translation": translation}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
