import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from app.model import SimpleCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("model/deepfake_model.pt", map_location=device))
model.to(device)
model.eval()

# SAME transform as training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Class labels (VERY IMPORTANT: must match training order)
class_names = ['Fake', 'Real']

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = class_names[predicted.item()]
    conf = confidence.item() * 100

    return label, conf


if __name__ == "__main__":
    path = input("Enter image path: ")
    label, conf = predict_image(path)
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2f}%")