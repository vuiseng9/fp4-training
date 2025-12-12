from pathlib import Path
import urllib.request

import torch
from torch.utils.data import Dataset


class TinyStories(Dataset):
    """
    TinyStories language modelling dataset, MNIST-style constructor.

    Args:
        root: directory to store/download TinyStories text files.
        train: if True, use TinyStories-train.txt, else TinyStories-valid.txt.
        download: if True, download the .txt file(s) if missing.
        tokenizer: object with `encode(str) -> List[int]` (e.g. ByteTokenizer).
        ctx_size: context length (number of tokens per sample).
        max_chars: optional cap on number of characters loaded from the file.

    Each item:
        x: ids[i : i+ctx_size]
        y: ids[i+1 : i+ctx_size+1]
    """

    # direct URLs to the plain .txt files (no HF Python deps)
    URLS = {
        "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
    }

    FILENAMES = {
        "train": "TinyStories-train.txt",
        "valid": "TinyStories-valid.txt",
    }

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        tokenizer=None,
        ctx_size: int = 128,
        max_chars: int | None = None,
    ):
        super().__init__()

        if tokenizer is None:
            raise ValueError("TinyStories: `tokenizer` must be provided")

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.train = train
        self.tokenizer = tokenizer
        self.ctx_size = ctx_size

        split = "train" if train else "valid"
        filename = self.FILENAMES[split]
        url = self.URLS[split]
        path = self.root / filename

        if download:
            self._download_if_needed(path, url)

        if not path.exists():
            raise FileNotFoundError(
                f"TinyStories: expected file {path} not found. "
                f"Set download=True to fetch it automatically."
            )

        # load text and optionally truncate
        text = path.read_text(encoding="utf-8")
        if max_chars is not None:
            text = text[:max_chars]

        # tokenize → 1D tensor of ids
        ids_list = self.tokenizer.encode(text)
        self.ids = torch.tensor(ids_list, dtype=torch.long)

        # ensure we have at least one full block (ctx_size)+ next token
        if len(self.ids) <= self.ctx_size:
            raise ValueError(
                f"TinyStories: not enough tokens ({len(self.ids)}) "
                f"for ctx_size={self.ctx_size}"
            )

    def _download_if_needed(self, path: Path, url: str):
        if path.exists():
            return
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)

    def __len__(self):
        # imagine ctx_size as a sliding window
        # number of samples is basically len(entire_text) - ctx_size
        return len(self.ids) - self.ctx_size

    def __getitem__(self, idx):
        # x: current block, y: next-token targets
        x = self.ids[idx : idx + self.ctx_size]
        y = self.ids[idx + 1 : idx + self.ctx_size + 1]
        return x, y
