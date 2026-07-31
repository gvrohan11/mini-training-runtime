import json, os, random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()

class JsonlLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")
    
    def log(self, **kw):
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()

    