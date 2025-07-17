import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

#
# 1. Pre‑processing pipeline
#
#   • Resize MNIST’s 28×28 images to 224×224 (ViT default)
#   • Replicate the single gray channel to 3 RGB channels
#   • Apply ImageNet mean / std normalisation
#
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#
# 2. MNIST test‑set loader
#
mnist_test = torchvision.datasets.MNIST(root=".", train=False,
                                        download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=64,
                         shuffle=False, num_workers=2, pin_memory=True)

#
# 3. Pre‑trained ViT‑B/16
#
weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
model   = torchvision.models.vit_b_16(weights=weights)
model.eval().cuda()          # move to GPU for speed (or .cpu() if you must)

#
# 4. Zero‑shot evaluation (no fine‑tuning)
#
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x.cuda())           # shape: [B, 1000]
        preds  = logits.argmax(dim=1)      # ImageNet label space (0‑999)
        #
        # MNIST digits are 0‑9; ImageNet classes are unrelated.
        # We simply test whether the numeric indices happen to match.
        #
        correct += (preds.cpu() == y).sum().item()
        total   += y.size(0)

print(f"Top‑1 accuracy (zero‑shot): {correct/total:.4%}")
