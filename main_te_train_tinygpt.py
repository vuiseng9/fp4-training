import os
os.environ["NVTE_NVFP4_DISABLE_RHT"] = "1" # Disable RHT in te nvfp4 recipe can work with fp32 training
os.environ["NVTE_NVFP4_DISABLE_2D_QUANTIZATION"] = "1"
import argparse
import math
from tqdm import tqdm
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import TinyGPT
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from dataset import TinyStories, ByteTokenizer

DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"

KEY2RECIPE = {
    "base": None, # use Transformer Engine module and use bf16 by default and --fp32
    "fp8": recipe.Float8CurrentScaling, # per tensor FP8
    "mxfp8": recipe.MXFP8BlockScaling,   # MXFP8
    "nvfp4": recipe.NVFP4BlockScaling,   # NVFP4
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyGPT on TinyStories with various Linear implementations")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size for training (default: 64), must be divisible by 32")
    parser.add_argument("-ep", "--epochs", type=int, default=20, help="Number of epochs to train (default: 3")
    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--recipe", type=str, default='base', choices=KEY2RECIPE.keys(), help=f"Linear layer implementation (required). Choices: {KEY2RECIPE.keys()}")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 precision (default: False, uses BF16 autocast per standard today)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Hyper-params ────────────────────────────────────────────────────────────
    BATCH_SIZE   = args.batch_size
    EPOCHS       = args.epochs
    LR           = args.lr
    DEVICE       = "cuda:0"
    CTX_SIZE     = 128

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
    tokenizer = ByteTokenizer()

    train_ds = TinyStories(root="raw_data", train=True,  download=True, tokenizer=tokenizer, ctx_size=CTX_SIZE, max_chars=2**16)
    test_ds  = TinyStories(root="raw_data", train=False, download=True, tokenizer=tokenizer, ctx_size=CTX_SIZE, max_chars=2**8)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, drop_last=True)

    print(f"Training #tokens/epoch: {len(train_loader) * BATCH_SIZE * CTX_SIZE:,}, Test #tokens: {len(test_loader) * BATCH_SIZE * CTX_SIZE:,}")

    # ── 3. Model ───────────────────────────────────────────────────────────────────

    model = TinyGPT(linear_impl="te",
                    vocab_size=tokenizer.VOCAB_SIZE,
                    ctx_size=CTX_SIZE,
                    embed_dim=128,
                    num_heads=4,
                    mlp_ratio=4.0).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(model)

    print(f"\n{str(args).replace('Namespace', 'Script Args')}\n")

    if DEBUG_MODE:
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"{n:30s} | {str(list(p.shape)):20s} | {str(p.numel()):10s}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable #params: {trainable_params:,}")

    def generate(prompt="Once upon a time, in a land far, far away, "):
        input_seq = torch.tensor(tokenizer(prompt)['input_ids']).to(DEVICE)
        generated_ids = model.generate(input_seq, max_new_tokens=64, do_sample=True)
        generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
        return f"| prompt | {prompt} \n| cont.  | {generated_text[len(prompt):]}"
    
    # ── 4. Training loop ───────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for n, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            with get_autocast_ctx():
                with get_te_ctx():
                    logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.VOCAB_SIZE), y.view(-1)) # essentially each position is a separate prediction.

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = (n * train_loss + loss.detach().item()) / (n + 1)
        train_ppl  = math.exp(train_loss)

        # ── 5. Quick eval ─────────────────────────────────────────────────────────
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(DEVICE), y.to(DEVICE)              
                with get_autocast_ctx():
                    with get_te_ctx():
                        logits = model(x)
                loss = criterion(logits.view(-1, tokenizer.VOCAB_SIZE), y.view(-1)) # essentially each position is a separate prediction.
                
                test_loss = (i * test_loss + loss.detach().item()) / (i + 1)
            
            test_ppl = math.exp(test_loss)
            test_bpc = test_loss / math.log(2)

        print(f"[Epoch {epoch}/{pbar.format_dict['elapsed']:5.1f} s] train_loss={train_loss:.4f} "
              f"train_ppl={train_ppl:.2f}  test_ppl={test_ppl:.2f} test_bpc={test_bpc:.4f}")

        if DEBUG_MODE:
            print(f"\n{generate()}\n")

    print("Training Done.")

    print(f"\n{generate()}\n")

if __name__ == "__main__":
    main()

