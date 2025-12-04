import argparse
from tqdm import tqdm
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import TinyViT
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

KEY2RECIPE = {
    "base": None, # use Transformer Engine module and use bf16 by default and --fp32
    "fp8": recipe.Float8CurrentScaling, # per tensor FP8
    "mxfp8": recipe.MXFP8BlockScaling,   # MXFP8
    "nvfp4": recipe.NVFP4BlockScaling,   # NVFP4
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyViT on MNIST with FP32")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size for training (default: 64), must be divisible by 32")
    parser.add_argument("-ep", "--epochs", type=int, default=3, help="Number of epochs to train (default: 3")
    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--recipe", type=str, default='base', choices=KEY2RECIPE.keys(), help=f"Linear layer implementation (required). Choices: {KEY2RECIPE.keys()}")
    # parser.add_argument("--impl", type=str, required=True, choices=REF_IMPL, help=f"Linear layer implementation (required). Choices: {REF_IMPL}")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 precision (default: False, uses BF16 autocast per standard today)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Hyper-params ────────────────────────────────────────────────────────────
    BATCH_SIZE   = args.batch_size
    EPOCHS       = args.epochs
    LR           = args.lr
    DEVICE       = "cuda:0"

    if BATCH_SIZE % 32 != 0:
        raise ValueError("Batch size must be divisible by 32 for 1D block quantization.")

    if args.recipe not in KEY2RECIPE:
        raise ValueError(f"Unsupported recipe {args.recipe}. Supported recipes: {KEY2RECIPE.keys()}")

    if args.recipe == 'nvfp4' and args.fp32:
        raise ValueError("NVFP4 does not support FP32 mode. RHT only support bfloat16")
    
    te_recipe = KEY2RECIPE[args.recipe]() if args.recipe != 'base' else None

    def get_autocast_ctx():
        if args.fp32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    
    def get_te_ctx():
        if te_recipe is None:
            return nullcontext()
        return te.fp8_autocast(fp8_recipe=te_recipe)
    
    # ── 2. Data ────────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),                      # (0,1) range, tensor shape (C,H,W)
        transforms.Normalize((0.1307,), (0.3081,))  # mean & std of MNIST
    ])

    train_ds = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, drop_last=True)

    # ── 3. Model ───────────────────────────────────────────────────────────────────

    model = TinyViT(linear_impl="te").to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(model)
    print(f"\n{str(args).replace('Namespace', 'Script Args')}\n")

    # ── 4. Training loop ───────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with get_autocast_ctx():
                with get_te_ctx():
                    logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().item() * y.size(0)
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
                with get_autocast_ctx():
                    with get_te_ctx():
                        preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        test_acc = 100.0 * correct / total

        print(f"[Epoch {epoch}/{pbar.format_dict['elapsed']:5.1f} s] train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%")

    print("Done.")


if __name__ == "__main__":
    main()

