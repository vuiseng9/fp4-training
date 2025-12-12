class ByteTokenizer:
    """
    Absolutely minimal byte-level tokenizer.
    - Each UTF-8 byte (0–255) is a token ID.
    - No PAD, BOS, EOS, or masks.
    """
    VOCAB_SIZE = 256  # 2^8 possible byte values

    def __init__(self):
        self.vocab_size = self.VOCAB_SIZE

    def encode(self, s: str):
        return list(s.encode("utf-8"))

    def decode(self, ids):
        # ids: list[int] in [0, 255]
        return bytes(ids).decode("utf-8", errors="ignore")

    def __call__(self, texts):
        """
        HF-ish style: accepts a single string or list of strings.
        Returns a dict with 'input_ids' as a list of lists.
        No padding, no truncation, no attention_mask.
        """
        if isinstance(texts, str):
            texts = [texts]
        batch_ids = [self.encode(t) for t in texts]
        return {"input_ids": batch_ids}

    def batch_decode(self, batch_ids):
        return [self.decode(ids) for ids in batch_ids]

if __name__ == "__main__":
    tokenizer = ByteTokenizer()

    texts = [
        "hello world",
        "a quick brown fox jumps over the lazy dog!",
    ]

    batch = tokenizer(texts)
    print("Tokenized ids:", batch["input_ids"])

    print("Decoded:", tokenizer.decode(batch["input_ids"][0]))

    