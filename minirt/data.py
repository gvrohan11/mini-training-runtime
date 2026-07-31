import numpy as np
import torch

def make_corpus(vocab_size, n_tokens, seed, concentration=0.1):
    rng = np.random.default_rng(seed)
    cdf = rng.dirichlet(np.full(vocab_size, concentration), size=vocab_size).cumsum(1)
    u = rng.random(n_tokens)
    tokens = np.empty(n_tokens, dtype=np.int64)
    cur = 0
    for i in range(n_tokens):
        tokens[i] = cur
        cur = int(np.searchsorted(cdf[cur], u[i]))
    return tokens

class Batcher:
    def __init__(self, tokens, batch_size, seq_len, seed, rank=0, world_size=1):
        self.tokens = torch.from_numpy(tokens)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        n_windows = len(tokens) - seq_len
        self.order = np.random.default_rng(seed).permutation(n_windows)
        self.pos = 0
    
    def next_batch(self):
        need = self.batch_size * self.world_size
        if self.pos + need > len(self.order):
            self.pos = 0
        chunk = self.order[self.pos : self.pos + need]
        self.pos += need
        mine = chunk[self.rank :: self.world_size][: self.batch_size]
        x = torch.stack([self.tokens[i : i + self.seq_len] for i in mine])
        y = torch.stack([self.tokens[i + 1 : i + 1 + self.seq_len] for i in mine])
        return x, y
    
    def state_dict(self):
        return {"pos": self.pos}
    
    def load_state_dict(self, sd):
        self.pos = sd["pos"]