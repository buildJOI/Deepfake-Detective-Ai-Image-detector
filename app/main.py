import io
import os
import json
import tempfile
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SusAI - AI Media Detector",
    description="Detect AI-generated / deepfake images and videos.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH       = os.getenv("MODEL_PATH", "model/deepfake_detector.keras")
CLASS_INDEX_PATH = os.getenv("CLASS_INDEX_PATH", "model/class_indices.json")

MAX_IMAGE_MB = 30
MAX_VIDEO_MB = 300

# All common image formats — including HEIC/HEIF from iPhone, WebP, etc.
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
    "image/tiff",
    "image/heic",
    "image/heif",
    "image/x-heic",
    "image/x-heif",
    "application/octet-stream",  # fallback for some Android phones
}

# All common video formats from Android/iPhone
ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/x-msvideo",       # .avi
    "video/mpeg",
    "video/quicktime",       # .mov (iPhone)
    "video/webm",
    "video/x-matroska",      # .mkv
    "video/3gpp",            # .3gp (older Android)
    "video/3gpp2",           # .3g2
    "video/x-m4v",           # .m4v
    "video/x-flv",           # .flv
    "application/octet-stream",  # fallback
}

# ---------------------------------------------------------------------------
# Load model + class indices at startup
# ---------------------------------------------------------------------------

print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("Model loaded!")

with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)

idx_to_label = {v: k.upper() for k, v in class_indices.items()}
print("Class mapping:", idx_to_label)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Resize, convert to RGB, rescale to [0,1] — matches training."""
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_pil(pil_img: Image.Image) -> dict:
    """Run inference on a PIL image."""
    arr = preprocess_image(pil_img)
    prob = float(model.predict(arr, verbose=0)[0][0])

    predicted_idx = 1 if prob > 0.51 else 0
    label = idx_to_label[predicted_idx]
    confidence = prob if predicted_idx == 1 else 1.0 - prob

    # If confidence below 51% → force FAKE
    if confidence * 100 < 51:
        label = "FAKE"
        predicted_idx = 0

    return {
        "prediction": predicted_idx,
        "label": label,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(prob, 4),
    }


def open_image_bytes(contents: bytes) -> Image.Image:
    """Try multiple methods to open image bytes — handles HEIC and unusual formats."""
    # Method 1: PIL directly
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        pass

    # Method 2: OpenCV decode
    try:
        np_arr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        pass

    raise ValueError("Could not decode image. File may be corrupt or unsupported format.")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "SusAI API is running", "version": "4.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "class_mapping": idx_to_label,
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict whether an image is AI-generated."""
    print(f"[IMAGE] Received: {file.filename} | type: {file.content_type}")

    contents = await file.read()

    if len(contents) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_IMAGE_MB} MB.")

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        pil_img = open_image_bytes(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = predict_pil(pil_img)
    result["filename"] = file.filename or "image"
    return result


# Legacy alias
@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    return await predict_image(file)


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    frames_every_n: int = 15,
    max_frames: int = 20,
):
    """Predict whether a video is AI-generated / deepfake."""
    print(f"[VIDEO] Received: {file.filename} | type: {file.content_type}")

    contents = await file.read()

    if len(contents) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_VIDEO_MB} MB.")

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # Determine file extension
    filename = file.filename or "video.mp4"
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        # Guess from content type
        ct = (file.content_type or "").lower()
        if "quicktime" in ct or "mov" in ct:
            ext = ".mov"
        elif "3gpp" in ct:
            ext = ".3gp"
        elif "webm" in ct:
            ext = ".webm"
        else:
            ext = ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file. Format may be unsupported.")

        frames = []
        frame_idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frames_every_n == 0:
                frames.append(frame)
            frame_idx += 1

        cap.release()
    finally:
        os.unlink(tmp_path)

    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames from video.")

    # Per-frame predictions
    frame_results = []
    fake_confidences = []
    real_confidences = []

    for i, frame_bgr in enumerate(frames):
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        r = predict_pil(pil_img)
        frame_results.append({"frame_index": i, **r})
        fake_confidences.append(r["confidence"] if r["label"] == "FAKE" else 100 - r["confidence"])
        real_confidences.append(r["confidence"] if r["label"] == "REAL" else 100 - r["confidence"])

    # Majority-vote verdict
    fake_votes = sum(1 for r in frame_results if r["label"] == "FAKE")
    real_votes = len(frame_results) - fake_votes
    verdict = "FAKE" if fake_votes >= real_votes else "REAL"

    return {
        "filename": filename,
        "verdict": verdict,
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "frames_analyzed": len(frame_results),
        "avg_fake_confidence": round(sum(fake_confidences) / len(fake_confidences), 2),
        "avg_real_confidence": round(sum(real_confidences) / len(real_confidences), 2),
        "processing_seconds": round(time.time() - t0, 2),
        "frame_details": frame_results,
    }