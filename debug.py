import argparse
import os
from tqdm import tqdm
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import TinyViT
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from models import REF_IMPL, LINEAR_IMPL

os.environ["NVTE_NVFP4_DISABLE_RHT"] = "1" # Disable RHT in te nvfp4 recipe 


KEY2RECIPE = {
    "base": None, # use Transformer Engine module and use bf16 by default and --fp32
    "fp8": recipe.Float8CurrentScaling, # per tensor FP8
    "mxfp8": recipe.MXFP8BlockScaling,   # MXFP8
    "nvfp4": recipe.NVFP4BlockScaling,   # NVFP4
}
KEY2REF_IMPL = {
    "base": "cublaslt",
    "fp8": "cublaslt_mxfp8",
    "mxfp8": "cublaslt_mxfp8",
    "nvfp4": "cublaslt_nvfp4",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyViT on MNIST with FP32")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size for training (default: 64), must be divisible by 32")
    parser.add_argument("-ep", "--epochs", type=int, default=3, help="Number of epochs to train (default: 3")
    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--recipe", type=str, default='base', choices=KEY2RECIPE.keys(), help=f"Linear layer implementation (required). Choices: {KEY2RECIPE.keys()}")
    # parser.add_argument("--impl", type=str, required=True, choices=REF_IMPL, help=f"Linear layer implementation (required). Choices: {REF_IMPL}")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 precision (default: False, uses BF16 autocast per standard today)")
    parser.add_argument("--per_step_loss", action="store_true", help="print loss per step")
    parser.add_argument("--debug_linear", action="store_true", help="Use single linear layer implementation")
    parser.add_argument("--print_io_stats", action="store_true", help="Print input/output stats for each layer, only works when --debug_linear is set")
    return parser.parse_args()


def debug_model(linear_fn):
    NFEAT=28*28
    NCLS=10
    IC=1024
    OC=128
    model = nn.Sequential(
        nn.Linear(in_features=28*28, out_features=1024),
        linear_fn(in_features=1024, out_features=128),
        # nn.ReLU(),
        nn.Linear(in_features=128, out_features=10)
    )
    return model

def print_io_hook(module, input, output):
    """
    A forward hook that prints input/output shapes and stats.
    
    Args:
        module: The layer the hook is attached to
        input: A tuple of inputs to the layer (usually just one tensor)
        output: The output tensor of the layer
    """
    layer_name = module.__class__.__name__
    
    print(f"\n[DEBUG] {layer_name}")
    print("=" * 40)
    
    # --- Inspect Input ---
    # Input is always a tuple in forward hooks
    if isinstance(input, tuple) and len(input) > 0:
        x = input[0] # The main input tensor
        print(f"Input : {x.dtype} | {x.float().max().item():10.6f} max | {x.float().mean().item():10.6f} mean | {x.float().std().item():10.6f} std")
        # Print first few elements for sanity check
        # print(f"  Slice: {x.flatten()[:5].tolist()}...")
    else:
        print(f"Input : {input}")

    # --- Inspect Output ---
    print(f"Output: {output.dtype} | {output.float().max().item():10.6f} max | {output.float().mean().item():10.6f} mean | {output.float().std().item():10.6f} std")
    # print(f"  Slice: {output.flatten()[:5].tolist()}...")
    print("=" * 40)


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
    if args.debug_linear:
        model = debug_model(te.Linear)
    else:
        model = TinyViT(linear_impl="te").to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    if args.debug_linear:
        model_b = debug_model(LINEAR_IMPL[KEY2REF_IMPL[args.recipe]])
    else:
        model_b = TinyViT(linear_impl=KEY2REF_IMPL[args.recipe]).to(DEVICE)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=LR)
    criterion_b = nn.CrossEntropyLoss()

    model = model.to(DEVICE)
    model_b = model_b.to(DEVICE)

    print(f"model: {model}")
    print(f"model_b: {model_b}")
    print(f"\n{str(args).replace('Namespace', 'Script Args')}\n")

    if args.print_io_stats and args.debug_linear:
            model[1].register_forward_hook(print_io_hook)
            model_b[1].register_forward_hook(print_io_hook)

    # ── 4. Training loop ───────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model_b.train()
        total, correct, loss_sum = 0, 0, 0.0
        total_b, correct_b, loss_sum_b = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if args.debug_linear:
                x = x.view(x.size(0), -1)  # flatten for single linear layer

            with get_autocast_ctx():
                logits_b = model_b(x)
                with get_te_ctx():
                    logits = model(x)
            loss = criterion(logits, y)
            loss_b = criterion_b(logits_b, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_b.zero_grad()
            loss_b.backward()
            optimizer_b.step()

            loss_sum += loss.detach().item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

            loss_sum_b += loss_b.detach().item() * y.size(0)
            preds_b = logits_b.argmax(1)
            correct_b += (preds_b == y).sum().item()
            total_b   += y.size(0)
            loss_delta = abs(loss.detach().item() - loss_b.detach().item())
            if args.per_step_loss:
                print(f"t{pbar.n} loss=       {loss.detach().item():.6f}")
                print(f"t{pbar.n} loss_b=     {loss_b.detach().item():.6f}")
                print(f"t{pbar.n} loss_delta= {loss_delta:.6f}\n")
                print("")
        train_acc = 100.0 * correct / total
        train_loss = loss_sum / total

        train_acc_b = 100.0 * correct_b / total_b
        train_loss_b = loss_sum_b / total_b

        # ── 5. Quick eval ─────────────────────────────────────────────────────────
        model.eval()
        model_b.eval()
        total, correct = 0, 0
        total_b, correct_b = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                if args.debug_linear:
                    x = x.view(x.size(0), -1)  # flatten for single linear layer

                with get_autocast_ctx():
                    preds_b = model_b(x).argmax(1)
                    with get_te_ctx():
                        preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                correct_b += (preds_b == y).sum().item()
                total   += y.size(0)
                total_b += y.size(0)
        test_acc = 100.0 * correct / total
        test_acc_b = 100.0 * correct_b / total_b

        print(f"\n[Epoch {epoch}/{pbar.format_dict['elapsed']:5.1f} s] train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%")
        print(f"[Epoch {epoch}/{pbar.format_dict['elapsed']:5.1f} s] train_loss_b={train_loss_b:.4f} "
                f"train_acc_b={train_acc_b:.2f}%  test_acc_b={test_acc_b:.2f}%\n")

    print("Done.")


if __name__ == "__main__":
    main()

