import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import wandb
from collections import Counter

from inference.task_model import TaskInferenceModel

# Config.
DATASET_PATH = Path("data/dataset/dataset.json")
SAVE_DIR = Path("data/models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES   = 3
HIDDEN_DIM  = 128
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
class TaskDataset(Dataset):
    def __init__(self, samples: list):
        self.X = torch.tensor(
            [s["features"] for s in samples], dtype=torch.float32
        )
        self.y = torch.tensor(
            [s["label"] for s in samples], dtype=torch.long
        )

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def get_class_weights(samples:list) -> torch.Tensor:
    """Compute inverse frequency weights for class imbalance."""
    counts = Counter(s["label"] for s in samples)
    total = len(samples)
    weights = [total / counts[i] for i in range(N_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32)

def get_sampler(samples: list) -> WeightedRandomSampler:
    """Oversample minority classes."""
    counts = Counter(s["label"] for s in samples)
    weights = [1.0 / counts[s["label"]] for s in samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# Training
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def main():
    # load data
    with open(DATASET_PATH) as f:
        data = json.load(f)
    samples = data["samples"]
    idx2action = data["idx2action"]
    N_FEATURES = data["n_features"]

    # train/val split
    train_samples, val_samples = train_test_split(
        samples, test_size=0.2, random_state=42,
        stratify=[s["label"] for s in samples]
    )
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = TaskDataset(train_samples)
    val_ds = TaskDataset(val_samples)

    sampler = get_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = TaskInferenceModel(N_FEATURES, N_CLASSES, HIDDEN_DIM).to(DEVICE)
    class_w = get_class_weights(train_samples).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # wandb
    wandb.init(project="proactive-home-ai", name="task-inference-v1", config={
        "n_features": N_FEATURES, "n_classes": N_CLASSES,
        "hidden_dim": HIDDEN_DIM, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "lr": LR,
    })

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, preds, labels = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")

    # final evaluation
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        labels, preds,
        target_names=[idx2action[str(i)] for i in range(N_CLASSES)]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    wandb.finish()


if __name__ == "__main__":
    main()