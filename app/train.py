import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

try:
    from app.model import EfficientNetDetector, SimpleCNN
except ModuleNotFoundError:
    from model import EfficientNetDetector, SimpleCNN

# ---------------------------------------------------------------------------
# Resolve project root regardless of where the script is launched from
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = os.getenv("DATA_DIR",  os.path.join(PROJECT_ROOT, "dataset", "train"))
MODEL_OUT   = os.getenv("MODEL_OUT", os.path.join(PROJECT_ROOT, "model", "deepfake_model.pt"))
MODEL_TYPE  = os.getenv("MODEL_TYPE", "efficientnet")
# Two-phase epochs: Phase 1 = frozen backbone, Phase 2 = full fine-tune
PHASE1_EPOCHS = int(os.getenv("PHASE1_EPOCHS", "30"))
PHASE2_EPOCHS = int(os.getenv("PHASE2_EPOCHS", "20"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "16"))
IMG_SIZE    = 224

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(MODEL_OUT) or ".", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

# ---------------------------------------------------------------------------
# Transforms — aggressive augmentation for small datasets
# ---------------------------------------------------------------------------

train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Dataset — separate dataset objects so train/val get different transforms
# ---------------------------------------------------------------------------

# Use ALL available images for training — with only ~120 images, holding
# out a validation split wastes precious training data.
train_dataset_full = datasets.ImageFolder(DATA_DIR, transform=train_tfm)

print(f"Device     : {device}")
print(f"Classes    : {train_dataset_full.classes}")
print(f"Total imgs : {len(train_dataset_full)}")
print(f"Train size : {len(train_dataset_full)}  (all data, no val split)")

train_loader = DataLoader(train_dataset_full, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_epoch(loader, mdl, opt, training: bool) -> tuple[float, float]:
    mdl.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                opt.zero_grad()
            outputs = mdl(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                opt.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def train_phase(mdl, opt, scheduler, epochs, phase_name, best_train_acc):
    print(f"\n{'='*50}")
    print(f"{phase_name} — {epochs} epochs")
    print(f"{'='*50}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(train_loader, mdl, opt, training=True)
        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
            f"({elapsed:.1f}s)"
        )
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(mdl.state_dict(), MODEL_OUT)
            print(f"  -> Saved best model (train_acc={train_acc*100:.2f}%)")
    return best_train_acc


# ---------------------------------------------------------------------------
# Main guard — required on Windows to prevent multiprocessing spawn crash
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    num_classes = len(train_dataset_full.classes)

    if MODEL_TYPE == "efficientnet":
        model = EfficientNetDetector(num_classes=num_classes, pretrained=True).to(device)
        print("Architecture: EfficientNet-B0 (pretrained)")

        # ------------------------------------------------------------------
        # Phase 1: freeze the entire backbone, train only the new head
        # This protects the pretrained features while the head adapts.
        # ------------------------------------------------------------------
        for param in model.model.features.parameters():
            param.requires_grad = False

        head_params = [p for p in model.parameters() if p.requires_grad]
        opt1 = optim.AdamW(head_params, lr=5e-3, weight_decay=1e-4)
        sch1 = CosineAnnealingLR(opt1, T_max=PHASE1_EPOCHS)

        best = 0.0
        best = train_phase(model, opt1, sch1, PHASE1_EPOCHS,
                           "Phase 1 — head only (backbone frozen)", best)

        # ------------------------------------------------------------------
        # Phase 2: unfreeze everything, fine-tune with a much lower LR
        # ------------------------------------------------------------------
        for param in model.parameters():
            param.requires_grad = True

        opt2 = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        sch2 = CosineAnnealingLR(opt2, T_max=PHASE2_EPOCHS)

        best = train_phase(model, opt2, sch2, PHASE2_EPOCHS,
                           "Phase 2 — full fine-tune (all layers)", best)

    else:
        model = SimpleCNN(num_classes=num_classes).to(device)
        print("Architecture: SimpleCNN")
        total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = CosineAnnealingLR(opt, T_max=total_epochs)
        best = train_phase(model, opt, sch, total_epochs, "Training", 0.0)

    # Always save the final state too
    final_out = MODEL_OUT.replace(".pt", "_final.pt")
    torch.save(model.state_dict(), final_out)

    print(f"\nTraining complete! Best train accuracy: {best*100:.2f}%")
    print(f"Best model : {MODEL_OUT}")
    print(f"Final model: {final_out}")