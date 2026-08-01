DDP = Distributed Data Parallel Processing

Idea: we take a bunch of training data, split it across multiple gpus, the gpus train on that data, then
the model averages the results of training across multiple gpus. Rather than train the same data for
all gpus, we split it amongst each gpu, making training go by faster.

# mini-training-runtime

A small, reproducible training loop for a byte-level GPT. The point is the runtime,
not the model: deterministic batching, seeded initialization, and append-only metrics
so two runs that differ only in a hyperparameter produce comparable curves.

## Layout

```
mini-training-runtime/
  minirt/
    __init__.py      # empty
    model.py         # TinyGPT
    data.py          # deterministic batcher
    utils.py         # seeding, logging
  train.py           # entry point
  hello_dist.py      # distributed smoke test
  runs/              # metrics output (gitignored)
  README.md
  requirements.txt
```

## Reproducibility

Batches are a pure function of `(seed, split, step)`: step 40 draws the same windows on
every run, machine, and process, regardless of global RNG state or how many batches came
before it. Evaluation draws from a step range far past the training stream, so adding an
eval pass never perturbs the training batches.

`set_seed` covers python, numpy, and torch. `--deterministic` additionally pins cuDNN and
cuBLAS to deterministic kernels for bit-identical runs on the same hardware, at some cost
in throughput.

## Output

Each run writes `runs/<run-name>/`:

- `config.json` — model config, data source, and the full arg namespace
- `metrics.jsonl` — one JSON object per logged step
- `ckpt.pt` — only with `--save-checkpoint`

```python
import pandas as pd
df = pd.read_json("runs/baseline/metrics.jsonl", lines=True)
df.dropna(subset=["val_loss"]).plot(x="step", y=["train_loss", "val_loss"])
```

JSONL is append-only and flushed per step, so an interrupted run still leaves everything
logged up to the kill.

## Model

`TinyGPT` is a pre-norm decoder-only transformer: learned token and position embeddings,
causal attention via `scaled_dot_product_attention`, GELU MLP, tied input/output
embeddings, and GPT-2 scaled init on residual projections. Vocabulary is raw bytes
(256 tokens), so any UTF-8 file works with no tokenizer to train or ship.

Defaults — 4 layers, 4 heads, 128-dim, 128-token context — are ~0.8M non-embedding
parameters and train in a couple of minutes on CPU.


## Status

| Milestone | Description | State |
|---|---|---|
| M0 | Single-process baseline | done |
| M1 | Distributed hello-world | done |
| M2 | Naive DDP (per-parameter all-reduce) | done |
| M3 | GPU port + scaling efficiency baseline | |
| M4 | Gradient bucketing | |
| M5 | Compute/comm overlap | |
| M6 | Deterministic checkpoint/resume | |
| M7 | Profiling pass + benchmark table | |


## M0 - Single-process baseline

Reference implementation and correctness oracle for everything that follows.

- `TinyGPT`: causal transformer (4 layers, 4 heads, d_model 128, vocab 256),
  written from scratch on `F.scaled_dot_product_attention`.
- Synthetic corpus from a first-order Markov source with Dirichlet-sampled
  transitions. Structured enough that loss falls well below `ln(256) = 5.545`,
  which proves the model is learning rather than memorizing noise.
- `Batcher`: deterministic index-permutation sampler. Data order is a pure
  function of `(seed, pos)`, and `pos` is a single integer - this is what makes
  bit-exact checkpoint resume possible in M6.
- Throughput measured after a warmup window, excluding startup and allocator
  warmup from the timing.

Result (M2 Pro, CPU, batch 32, seq 128):

| Steps | Start loss | End loss | Throughput |
|---|---|---|---|
| 200 | 5.64 | 4.78 | ~60k tok/s |
| 2000 | 5.64 | 3.70 | ~60k tok/s |

Random-guess baseline is `ln(vocab_size) = 5.545`.

Summary: 
Built the reference training run: a 4-layer transformer (TinyGPT), a synthetic corpus with
learnable structure, a deterministic batcher, and a loop that logs loss and tokens/sec after
a warmup window

Result: loss fell 5.64 → 4.78 over 200 steps, and → 3.70 over 2000. Random guessing would sit at
5.545, so the model is genuinely learning. ~60–68k tok/s on your M2 Pro CPU. This run is the correctness oracle for everything after it.

## M1 - Distributed hello-world

Minimal verification that collective communication works before any training
code depends on it.

Each rank builds a one-element tensor holding its own rank and calls
`all_reduce` with the default SUM op. Every rank should converge on
`N(N-1)/2`.

    ./run_dist.sh 2 hello_dist.py   # all ranks -> 1.0
    ./run_dist.sh 4 hello_dist.py   # all ranks -> 6.0

Two properties this establishes, both load-bearing later:

- `all_reduce` is **in-place**. M4's bucketing depends on this - flatten
  gradients into one contiguous buffer, reduce the buffer, copy back out.
- **Every rank ends with the same value.** Reduce *and* broadcast. This is why
  data-parallel training needs no parameter server: all ranks independently
  arrive at identical gradients, so `opt.step()` keeps their weights in sync.

Summary: start a process group, have each rank make a tensor holding its own rank number, all-reduce, print

Result: 2 ranks → both printed 1.0, 4 ranks → all printed 6.0. Confirmed the processes can find each other
and communicate. Also there's a macOS IPv6 rendezvous hang (startup handshake where separate processes find
each other and agree they're one job), which is why run_dist.sh exists.

### macOS note

`torchrun` hangs on rendezvous on macOS - c10d resolves the master address to
IPv6 loopback (`::1`), and the reverse lookup fails. `run_dist.sh` pins gloo to
`lo0`, forces an IPv4 literal master address, and disables the libuv TCPStore.

## M2 - Naive DDP

Data-parallel training written directly against `torch.distributed`. No
`DistributedDataParallel`.

Four pieces:

1. **Identical initialization** - `dist.broadcast(param.data, src=0)` for every
   parameter before training. Same-seed init would work here, but broadcasting
   is what makes correctness independent of seed alignment.
2. **Data sharding** - all ranks build the same permutation from the same seed,
   then each takes `[rank::world_size]` of every chunk. No rank sees another
   rank's samples.
3. **Gradient averaging** - after `backward()`, `all_reduce` each parameter's
   gradient and divide by `world_size`. The divide is not optional: `all_reduce`
   sums, and summed gradients mean an effective learning rate `world_size` times
   too large.
4. **Rank-0-only logging** - with the reported loss itself all-reduced, so the
   logged value is the global mean rather than one rank's local view.

### Correctness

The single-process run is the oracle. Same global batch, split two ways:

    python train.py --steps 200 --batch-size 64                # 1 rank  x 64
    ./run_dist.sh 2 train_ddp.py --steps 200 --batch-size 32   # 2 ranks x 32

| Step | 1 rank x 64 | 2 ranks x 32 |
|---|---|---|
| 10 | 5.6270 | 5.6270 |
| 100 | 5.0436 | 5.0436 |
| 200 | 4.5645 | 4.5645 |

Identical to four decimals at every logged step. Gradient averaging over two
shards of 32 is algebraically the mean over 64; at this model size the
reduction order happens to agree bit-for-bit too.

Note `--batch-size` is **per-rank** in `train_ddp.py`. Global batch is
`batch_size x world_size`.

### Throughput: the "before" number

| Config | Throughput |
|---|---|
| 1 rank x 64 | ~68k tok/s |
| 2 ranks x 32 | ~22k tok/s |

Twice the processes, a third of the throughput. Three causes, two of which are
the point of the next milestones:

- Two processes contending for one laptop's cores at `OMP_NUM_THREADS=1`
  (an artifact of local CPU testing; disappears on real multi-GPU hardware).
- **~30 separate `all_reduce` calls per step**, one per parameter tensor. Each
  carries fixed latency overhead, and this model is small enough that the
  overhead dominates the bytes moved. -> M4 (bucketing).
- **Communication runs after backward completes.** Compute sits idle waiting
  for the sync. -> M5 (overlap).

This is the baseline the rest of the project is measured against.

Summary: Wrote data-parallel training from scratch: broadcast weights from rank 0, shard the data by rank, then after each backward pass all-reduce every gradient and divide by world size. No DistributedDataParallel.

2 Results, 1 good and 1 bad:

Correctness: 2 ranks × batch 32 produced identical loss to four decimals against 1 rank × batch 64, at every logged step. Your DDP is mathematically exact.

Throughput: 68k → 22k tok/s. Slower with more processes. Three causes — CPU contention (a local-testing artifact), ~30 separate all-reduce calls per step where latency dominates, and communication running only after backward finishes while compute sits idle.

## Setup

    conda create -n mtr python=3.12 -y
    conda activate mtr
    python -m pip install -r requirements.txt

    python train.py --steps 2000          # single process
    ./run_dist.sh 2 hello_dist.py         # distributed smoke test