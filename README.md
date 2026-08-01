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
| M2 | Naive DDP (per-parameter all-reduce) | in progress |
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


### macOS note

`torchrun` hangs on rendezvous on macOS - c10d resolves the master address to
IPv6 loopback (`::1`), and the reverse lookup fails. `run_dist.sh` pins gloo to
`lo0`, forces an IPv4 literal master address, and disables the libuv TCPStore.

## Setup

    conda create -n mtr python=3.12 -y
    conda activate mtr
    python -m pip install -r requirements.txt

    python train.py --steps 2000          # single process
    ./run_dist.sh 2 hello_dist.py         # distributed smoke test