import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from app.model import SimpleCNN

# -----------------------------
# Setup
# -----------------------------

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset path
DATA_DIR = "dataset/train"

if not os.path.exists(DATA_DIR):
    raise Exception(f"Dataset path not found: {DATA_DIR}")

# -----------------------------
# Transforms (MUST match inference)
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # smaller = faster on CPU
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load Dataset
# -----------------------------

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("CLASS ORDER:", dataset.classes)
print("Total images (original):", len(dataset))

# 🔥 LIMIT DATASET SIZE FOR FAST TESTING
subset_size = 5000   # Change to 2000 if still slow
dataset = Subset(dataset, range(min(subset_size, len(dataset))))

print("Total images (after limit):", len(dataset))

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# Model
# -----------------------------

model = SimpleCNN(num_classes=len(dataset.dataset.classes)).to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training Loop
# -----------------------------

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Save every 100 batches
        if batch_idx % 100 == 0:
            torch.save(model.state_dict(), "model/temp_model.pt")
            print(f"Saved checkpoint at batch {batch_idx}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")

# -----------------------------
# Final Save
# -----------------------------

torch.save(model.state_dict(), "model/deepfake_model.pt")

print("✅ Training complete!")