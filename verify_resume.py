import os
import subprocess
import sys

import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(ROOT, "runs")
os.makedirs(RUN_DIR, exist_ok=True)


def run(cmd):
    subprocess.run(cmd, check=True, cwd=ROOT)


def load_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model"], ckpt["optimizer"]


def deep_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    ref_path = os.path.join(RUN_DIR, "ref.ckpt")
    part1_path = os.path.join(RUN_DIR, "part1.ckpt")
    part2_path = os.path.join(RUN_DIR, "part2.ckpt")

    for p in [ref_path, part1_path, part2_path]:
        if os.path.exists(p):
            os.remove(p)

    run([sys.executable, "train_ddp.py", "--steps", "100", "--run-name", "ref", "--checkpoint-every", "100", "--checkpoint-dir", "runs"])
    run([sys.executable, "train_ddp.py", "--steps", "50", "--run-name", "part1", "--checkpoint-every", "50", "--checkpoint-dir", "runs"])
    run([sys.executable, "train_ddp.py", "--steps", "100", "--run-name", "part2", "--resume", "runs/part1.ckpt", "--checkpoint-every", "100", "--checkpoint-dir", "runs"])

    ref_model, ref_opt = load_state(ref_path)
    part2_model, part2_opt = load_state(part2_path)

    model_ok = deep_equal(ref_model, part2_model)
    opt_ok = deep_equal(ref_opt, part2_opt)

    if model_ok and opt_ok:
        print("PASS")
        return 0

    print("FAIL")
    if not model_ok:
        print("model state differs")
    if not opt_ok:
        print("optimizer state differs")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
