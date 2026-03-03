from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from app.model import load_model

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Deepfake API is running 🚀"}


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
MODEL_PATH = "model/deepfake_model.pt"
model = load_model(MODEL_PATH, device)

# MUST match training transform
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# VERY IMPORTANT
idx_to_label = {
    0: "FAKE",
    1: "REAL"
}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tfm(Image.fromarray(img)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "prediction": int(pred),
        "label": idx_to_label[pred],
        "confidence": round(confidence * 100, 2)
    }