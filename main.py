# main.py (API-only, drop-in replacement)
import os
# reduce TF verbosity BEFORE importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import numpy as np
import cv2
import asyncio
from asyncio import Semaphore

app = FastAPI(title="AI Image Detector - API only")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL ----------------
MODEL_PATH = "best_model.keras"
model = None
app.state.model_error = None

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    # keep server up but show error in / and /health
    app.state.model_error = str(e)
    model = None

# Limit concurrent inference requests
semaphore = Semaphore(2)

# ---------------- STARTUP WARMUP ----------------
@app.on_event("startup")
def warmup_model():
    if model is None:
        return
    try:
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        # run in threadpool so startup loop isn't blocked
        import asyncio as _asyncio
        loop = _asyncio.get_event_loop()
        loop.run_until_complete(run_in_threadpool(model.predict, dummy))
    except Exception:
        # ignore warmup errors; model load error already captured if any
        pass

# ---------------- ROOT (JSON health) ----------------
@app.get("/", response_class=JSONResponse)
def root():
    """
    API-only root. Platforms call '/' so return JSON status here.
    """
    if app.state.model_error:
        return JSONResponse(status_code=500, content={"status": "error", "model_error": app.state.model_error})
    return {"status": "ok"}

# ---------------- HEALTH ----------------
@app.get("/health", response_class=JSONResponse)
def health():
    if app.state.model_error:
        return JSONResponse(status_code=500, content={"status": "error", "model_error": app.state.model_error})
    return {"status": "ok"}

# ---------------- IMAGE PREP ----------------
def prepare_image(img_bytes: bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_cv is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    return preprocess_input(img_array)

# ---------------- INFERENCE LOGIC ----------------
async def run_prediction(processed_img):
    if model is None:
        raise RuntimeError("Model not loaded")
    pred = await run_in_threadpool(model.predict, processed_img)
    # Safely extract a scalar from the model output
    try:
        return float(pred[0][0])
    except Exception:
        arr = np.array(pred).ravel()
        if arr.size:
            return float(arr[0])
        raise RuntimeError("Unexpected model output shape")

# ---------------- PREDICT ROUTE ----------------
@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    if app.state.model_error:
        raise HTTPException(status_code=500, detail=f"Model load error: {app.state.model_error}")

    try:
        async with semaphore:
            contents = await file.read()
            processed = prepare_image(contents)

            score = await asyncio.wait_for(
                run_prediction(processed),
                timeout=25
            )

        prob_real = score
        prob_ai = 1 - score
        TH = 0.60

        if prob_real > TH:
            label = "Real"
            confidence = prob_real
        elif prob_ai > TH:
            label = "AI Generated"
            confidence = prob_ai
        else:
            label = "Other Image"
            confidence = max(prob_real, prob_ai)

        return {
            "filename": file.filename,
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%"
        }

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Model inference timed out.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
