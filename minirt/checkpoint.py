import os
import random

import numpy as np
import torch


def save(path, model, opt, batcher, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "batcher": batcher.state_dict(),
        "step": step,
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load(path, model, opt, batcher):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["optimizer"])
    batcher.load_state_dict(checkpoint["batcher"])

    torch.set_rng_state(checkpoint["rng"]["torch"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    random.setstate(checkpoint["rng"]["python"])
    if torch.cuda.is_available() and checkpoint["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])

    return checkpoint["step"]
