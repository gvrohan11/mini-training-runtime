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

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py                                   # synthetic corpus, 500 steps
python train.py --data data/input.txt --steps 2000 --run-name baseline
python train.py --n-layer 6 --n-embd 256 --save-checkpoint --sample-tokens 400
```

`--device` resolves `auto` to cuda → mps → cpu. `python train.py --help` lists every flag.

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
