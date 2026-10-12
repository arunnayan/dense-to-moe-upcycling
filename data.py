"""TinyShakespeare, character level.

Everything here is deterministic: the split is fixed, training batches come
from a seeded generator, and validation always uses the SAME fixed batches,
so a jump in the val curve can only come from the model, not from sampling.
"""
import os
import urllib.request

import torch

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "input.txt")


def load_text():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, encoding="utf-8") as f:
        return f.read()


class CharData:
    def __init__(self, block_size=128, val_frac=0.1):
        text = load_text()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(ids) * (1 - val_frac))
        self.train_ids, self.val_ids = ids[:n], ids[n:]
        self.block_size = block_size

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def _batch(self, ids, batch_size, gen):
        ix = torch.randint(len(ids) - self.block_size - 1, (batch_size,), generator=gen)
        x = torch.stack([ids[i:i + self.block_size] for i in ix])
        y = torch.stack([ids[i + 1:i + 1 + self.block_size] for i in ix])
        return x, y

    def train_batch(self, batch_size, gen):
        return self._batch(self.train_ids, batch_size, gen)

    def fixed_val_batches(self, n_batches, batch_size, seed=1234):
        """Same list of val batches every time it is called with the same seed."""
        gen = torch.Generator().manual_seed(seed)
        return [self._batch(self.val_ids, batch_size, gen) for _ in range(n_batches)]
