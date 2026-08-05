import argparse, time
import torch
from minirt.data import Batcher, make_corpus
from minirt.model import TinyGPT
from minirt.utils import JsonlLogger, set_seed, sync

def main():
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
    batcher = Batcher(tokens, args.batch_size, args.seq_len, args.seed)
    model = TinyGPT(args.vocab_size, args.seq_len, args.d_model, args.n_head, args.n_layer).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log = JsonlLogger(f"runs/{args.run_name}.jsonl")

    t0, timed_tokens = None, 0
    last_t, last_tokens = None, 0
    for step in range(1, args.steps + 1):
        x, y = batcher.next_batch()
        x = x.to(dev)
        y = y.to(dev)

        if step == args.warmup_steps + 1:
            sync(dev)
            now = time.perf_counter()
            t0, timed_tokens = now, 0
            last_t, last_tokens = now, 0

        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        timed_tokens += x.numel()

        if step % args.log_every == 0:
            sync(dev)
            now = time.perf_counter()
            interval_tps = (timed_tokens - last_tokens) / (now - last_t) if last_t is not None and now > last_t else float("nan")
            cumulative_tps = timed_tokens / (now - t0) if t0 is not None and now > t0 else float("nan")
            print(f"step {step:4d} | loss {loss.item():.4f} | interval {interval_tps:8.0f} tok/s | cumulative {cumulative_tps:8.0f} tok/s")
            log.log(step=step, loss=loss.item(), interval_tps=interval_tps,
                    cumulative_tps=cumulative_tps, world_size=1, run=args.run_name)
            last_t, last_tokens = now, timed_tokens
            
    sync(dev)
    total = timed_tokens / (time.perf_counter() - t0) if t0 is not None else float("nan")
    print(f"\nfinal: {total:.0f} tok/s over {args.steps - args.warmup_steps} steps")



if __name__ == "__main__":
    main()