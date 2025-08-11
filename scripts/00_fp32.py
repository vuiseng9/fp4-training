import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vit import TinyViT


# ── 1. Hyper-params ────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = int(os.getenv("NEPOCH", 3))
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── 2. Data ────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),                      # (0,1) range, tensor shape (C,H,W)
    transforms.Normalize((0.1307,), (0.3081,))  # mean & std of MNIST
])

train_ds = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ── 3. Model ───────────────────────────────────────────────────────────────────

model = TinyViT().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print(model)
criterion = nn.CrossEntropyLoss()

# ── 4. Training loop ───────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)

    train_acc = 100.0 * correct / total
    train_loss = loss_sum / total

    # ── 5. Quick eval ─────────────────────────────────────────────────────────
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    test_acc = 100.0 * correct / total

    print(f"[Epoch {epoch}] train_loss={train_loss:.4f} "
          f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%")

print("Done.")
