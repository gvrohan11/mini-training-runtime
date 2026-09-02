import argparse
import contextlib
import os
import time

import torch
import torch.distributed as dist

from minirt.bucket import GradientBucketer
from minirt.checkpoint import load, save
from minirt.data import Batcher, make_corpus
from minirt.model import TinyGPT
from minirt.utils import JsonlLogger, set_seed, sync


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

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
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--comm-dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--zero-stage", type=int, default=0, choices=[0, 1])
    p.add_argument("--bucket-mb", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--checkpoint-dir", default="runs")
    p.add_argument("--resume", default=None)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-steps", type=int, default=20)
    p.add_argument("--run-name", default="baseline")
    args = p.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dev = f"cuda:{local_rank}"
    else:
        dev = args.device

    tokens = make_corpus(args.vocab_size, args.corpus_tokens, args.seed)
    batcher = Batcher(tokens, args.batch_size, args.seq_len, args.seed, rank=rank, world_size=world_size)
    model = TinyGPT(args.vocab_size, args.seq_len, args.d_model, args.n_head, args.n_layer).to(dev)

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    if args.zero_stage == 1:
        from minirt.zero import ZeroOptimizer

        opt = ZeroOptimizer(model.parameters(), lambda params: torch.optim.AdamW(params, lr=args.lr), rank=rank, world_size=world_size)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    comm_dtype = None if args.comm_dtype == "fp32" else torch.bfloat16
    bucketer = GradientBucketer(model.parameters(), bucket_mb=args.bucket_mb, comm_dtype=comm_dtype)
    bucketer.register_hooks()
    log = JsonlLogger(f"runs/{args.run_name}.jsonl") if is_main else None

    resume_step = 0
    if args.resume is not None:
        resume_step = load(args.resume, model, opt, batcher)
        if is_main:
            print(f"resumed from step {resume_step} using {args.resume}")

    profiler = None
    if args.profile and is_main:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=args.profile_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"runs/trace_{args.run_name}"),
            record_shapes=True,
            with_stack=True,
        )

    prof_ctx = profiler if profiler is not None else contextlib.nullcontext()

    t0, timed_tokens = None, 0
    last_t, last_tokens = None, 0
    try:
        with prof_ctx:
            for step in range(resume_step + 1, args.steps + 1):
                x, y = batcher.next_batch()
                x = x.to(dev)
                y = y.to(dev)

                if step == resume_step + args.warmup_steps + 1:
                    sync(dev)
                    now = time.perf_counter()
                    t0, timed_tokens = now, 0
                    last_t, last_tokens = now, 0

                if args.dtype == "bf16" and dev.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, loss = model(x, y)
                else:
                    _, loss = model(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                bucketer.wait_and_copy(world_size)
                opt.step()
                timed_tokens += x.numel()

                if step % args.log_every == 0:
                    sync(dev)
                    loss_value = loss.detach().clone()
                    dist.all_reduce(loss_value)
                    loss_value = loss_value / world_size

                    now = time.perf_counter()
                    interval_tps = (timed_tokens - last_tokens) / (now - last_t) if last_t is not None and now > last_t else float("nan")
                    cumulative_tps = timed_tokens / (now - t0) if t0 is not None and now > t0 else float("nan")
                    global_interval_tps = interval_tps * world_size
                    global_cumulative_tps = cumulative_tps * world_size

                    if is_main:
                        print(f"step {step:4d} | loss {loss_value.item():.4f} | interval {global_interval_tps:8.0f} tok/s | cumulative {global_cumulative_tps:8.0f} tok/s")
                        log.log(step=step, loss=loss_value.item(), interval_tps=global_interval_tps,
                                cumulative_tps=global_cumulative_tps, world_size=world_size, run=args.run_name)

                    last_t, last_tokens = now, timed_tokens

                if args.checkpoint_every and step % args.checkpoint_every == 0:
                    if is_main:
                        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name}.ckpt")
                        save(ckpt_path, model, opt, batcher, step, args)

                if profiler is not None:
                    profiler.step()

            sync(dev)
            if is_main:
                total = timed_tokens / (time.perf_counter() - t0) if t0 is not None else float("nan")
                global_total = total * world_size
                print(f"\nfinal: {global_total:.0f} tok/s over {args.steps - args.warmup_steps} steps")
                if args.checkpoint_every:
                    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name}.ckpt")
                    save(ckpt_path, model, opt, batcher, args.steps, args)

            peak_mem = None
            if dev.startswith("cuda"):
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                if is_main:
                    print(f"peak memory: {peak_mem:.2f} GiB")
                    if log is not None:
                        log.log(step=args.steps, loss=float("nan"), interval_tps=float("nan"), cumulative_tps=float("nan"),
                                world_size=world_size, run=args.run_name, peak_memory_gib=peak_mem, type="summary")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()