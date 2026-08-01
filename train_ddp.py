import argparse
import os
import time

import torch
import torch.distributed as dist

from minirt.data import Batcher, make_corpus
from minirt.model import TinyGPT
from minirt.utils import JsonlLogger, set_seed, sync


def main():
    dist.init_process_group(backend="gloo")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_main = rank == 0

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cpu")
    p.add_argument("--corpus-tokens", type=int, default=200_000)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--run-name", default="baseline")
    args = p.parse_args()

    set_seed(args.seed)
    dev = args.device

    tokens = make_corpus(args.vocab_size, args.corpus_tokens, args.seed)
    batcher = Batcher(tokens, args.batch_size, args.seq_len, args.seed, rank=rank, world_size=world_size)
    model = TinyGPT(args.vocab_size, args.seq_len, args.d_model, args.n_head, args.n_layer).to(dev)

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log = JsonlLogger(f"runs/{args.run_name}.jsonl") if is_main else None

    t0, timed_tokens = None, 0
    try:
        for step in range(1, args.steps + 1):
            x, y = batcher.next_batch()
            x = x.to(dev)
            y = y.to(dev)

            if step == args.warmup_steps + 1:
                sync(dev)
                t0, timed_tokens = time.perf_counter(), 0

            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad /= world_size

            opt.step()
            timed_tokens += x.numel()

            if step % args.log_every == 0:
                sync(dev)
                tps = timed_tokens / (time.perf_counter() - t0) if t0 is not None else float("nan")
                loss_value = loss.detach().clone()
                dist.all_reduce(loss_value)
                loss_value = loss_value / world_size

                if is_main:
                    print(f"step {step:4d} | loss {loss_value.item():.4f} | {tps:8.0f} tok/s")
                    log.log(step=step, loss=loss_value.item(), tokens_per_sec=tps,
                            world_size=world_size, run=args.run_name)

        sync(dev)
        if is_main:
            total = timed_tokens / (time.perf_counter() - t0) if t0 is not None else float("nan")
            print(f"\nfinal: {total:.0f} tok/s over {args.steps - args.warmup_steps} steps")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()