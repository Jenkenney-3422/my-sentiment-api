import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Record when the app started
start_time = time.time()

app = FastAPI(title="Sentiment API - Render Deploy")
templates = Jinja2Templates(directory="templates")

# --- CLOUD CONFIGURATION ---
# On Render, we don't use local paths or GPUs. 
# We let Hugging Face manage the cache.
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"

print("🧠 Loading model into Cloud Memory (CPU)...")
classifier = pipeline(
    "sentiment-analysis", 
    model=MODEL_NAME, 
    device=-1  # -1 forces CPU (required for Render Free Tier)
)

class SingleInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

@app.get("/health")
def health_check():
    """
    Quick endpoint to check if the service is awake 
    without running heavy AI models.
    """
    uptime = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return {
        "status": "healthy",
        "uptime": uptime,
        "memory_safety": "optimized"
    }
    
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_single(data: SingleInput):
    prediction = classifier(data.text)[0]
    return {"text": data.text, "result": prediction}

@app.post("/predict_batch")
async def predict_batch(data: BatchInput):
    predictions = classifier(data.texts)
    results = [{"text": t, "result": p} for t, p in zip(data.texts, predictions)]
    return {"batch_results": results}