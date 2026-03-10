import io
import os
import tempfile
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from app.model import load_model
from app.video_utils import sample_frames

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Image & Video Detector",
    description="Detect AI-generated / deepfake images and videos.",
    version="2.0.0",
)

# Allow Flutter (and any other) client to call this API
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "model/deepfake_model.pt")
MODEL_TYPE = os.getenv("MODEL_TYPE", "efficientnet")   # "simple" | "efficientnet"

# Max upload sizes
MAX_IMAGE_MB = 20
MAX_VIDEO_MB = 200

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/x-msvideo", "video/mpeg", "video/quicktime", "video/webm"}

IDX_TO_LABEL = {0: "FAKE", 1: "REAL"}

# ---------------------------------------------------------------------------
# Model + transform (loaded once at startup)
# ---------------------------------------------------------------------------

model = load_model(MODEL_PATH, DEVICE, model_type=MODEL_TYPE)

# All images are resized to 224x224 — MUST match training
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predict_pil(pil_img: Image.Image) -> dict:
    """Run inference on a single PIL image and return a result dict."""
    tensor = tfm(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return {
        "prediction": int(pred),
        "label": IDX_TO_LABEL[pred],
        "confidence": round(confidence * 100, 2),
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "AI Image & Video Detector API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_path": MODEL_PATH,
        "model_type": MODEL_TYPE,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict whether a single image is AI-generated / deepfake."""
    # File-type guard
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {sorted(ALLOWED_IMAGE_TYPES)}",
        )

    contents = await file.read()

    # Size guard
    if len(contents) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_IMAGE_MB} MB.")

    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. File may be corrupt.")

    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    result = _predict_pil(pil_img)
    result["filename"] = file.filename
    return result


# Keep the old /predict route as an alias so existing clients don't break
@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    return await predict_image(file)


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    frames_every_n: int = 15,
    max_frames: int = 20,
):
    """
    Predict whether a video is AI-generated / deepfake.

    Samples up to `max_frames` frames (1 every `frames_every_n` frames),
    runs per-frame inference, and returns an aggregated verdict.
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {sorted(ALLOWED_VIDEO_TYPES)}",
        )

    contents = await file.read()

    if len(contents) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_VIDEO_MB} MB.")

    # Write to a temp file so OpenCV can open it by path
    suffix = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        frames_bgr = sample_frames(tmp_path, every_n=frames_every_n, max_frames=max_frames)
    finally:
        os.unlink(tmp_path)

    if not frames_bgr:
        raise HTTPException(status_code=400, detail="Could not extract frames from video.")

    # Per-frame predictions
    frame_results = []
    fake_confidences = []
    real_confidences = []

    for i, frame in enumerate(frames_bgr):
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        r = _predict_pil(pil_img)
        frame_results.append({"frame_index": i, **r})
        fake_confidences.append(r["confidence"] if r["label"] == "FAKE" else 100 - r["confidence"])
        real_confidences.append(r["confidence"] if r["label"] == "REAL" else 100 - r["confidence"])

    # Majority-vote verdict
    fake_votes = sum(1 for r in frame_results if r["label"] == "FAKE")
    real_votes = len(frame_results) - fake_votes
    verdict = "FAKE" if fake_votes >= real_votes else "REAL"
    avg_fake_conf = round(sum(fake_confidences) / len(fake_confidences), 2)
    avg_real_conf = round(sum(real_confidences) / len(real_confidences), 2)

    return {
        "filename": file.filename,
        "verdict": verdict,
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "frames_analyzed": len(frame_results),
        "avg_fake_confidence": avg_fake_conf,
        "avg_real_confidence": avg_real_conf,
        "processing_seconds": round(time.time() - t0, 2),
        "frame_details": frame_results,
    }