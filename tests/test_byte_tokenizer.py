import pytest
from dataset import ByteTokenizer


@pytest.fixture
def tokenizer():
    return ByteTokenizer()


def test_single_text(tokenizer):
    text = "hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_batch_of_sequences(tokenizer):
    texts = [
        "hello world",
        "a quick brown fox jumps over the lazy dog!",
        "Python 3.10 is great!",
    ]
    
    batch = tokenizer(texts)
    decoded = tokenizer.batch_decode(batch["input_ids"])
    
    assert decoded == texts


def test_special_characters(tokenizer):
    text = "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_unicode_text(tokenizer):
    text = "Hello 世界 🌍"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 256
    assert ByteTokenizer.VOCAB_SIZE == 256

