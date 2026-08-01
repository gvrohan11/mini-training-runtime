## Processes and identity

Process — an independent running program with its own memory. Your 2-rank run is two OS processes; they share nothing except what they explicitly send over the network.

Thread — a unit of execution inside a process, sharing that process's memory. Python's GIL makes threads bad for CPU-bound work, which is exactly why distributed training uses processes, not threads. (OMP_NUM_THREADS=1 limits threads within each of your processes.)

Rank ✓ — a process's global ID, 0 to world_size−1.

Local rank ✓ — ID within one machine. On a 2-node × 8-GPU job, rank 11 has local_rank 3. This is what you use to pick a GPU: torch.cuda.set_device(local_rank).

World size ✓ — total process count across all machines.

Rendezvous ✓ — the startup handshake where processes find each other via a key-value store and agree on the topology.

Launcher ✓ — torchrun / torch.distributed.run. Spawns the processes and sets RANK/WORLD_SIZE/LOCAL_RANK in each one's environment.

## Communication

Collective ✓ — an operation every rank participates in. Miss one rank and you deadlock.

All-reduce ✓ — combine values across ranks (sum by default), give the result to everyone. The core primitive of data parallelism.

Broadcast ✓ — one rank's value copied to all.

Reduce-scatter — combine across ranks, but each rank keeps only its slice of the result. Half of ZeRO.

All-gather — each rank contributes a piece; everyone ends with the full concatenation. The other half of ZeRO. Note: reduce-scatter + all-gather = all-reduce.

Barrier — sync point; every rank waits until all arrive.

Backend ✓ — the transport implementing collectives. gloo for CPU, nccl for NVIDIA GPUs.

Ring all-reduce — the algorithm NCCL uses: data flows around a ring so bandwidth cost stays constant regardless of rank count. Worth understanding before interviews.

Parallelism strategies

Data parallelism (DP) ✓ — every rank holds a full model copy, processes different data, syncs gradients. What you built.

Shard — split one logical thing across ranks so each holds a piece. You shard data now; ZeRO shards optimizer state.

Tensor parallelism (TP) — split individual matmuls across GPUs. For when one layer doesn't fit on one device.

Pipeline parallelism (PP) — different layers on different GPUs, micro-batches flowing through stages.

ZeRO / FSDP — shard optimizer state, gradients, and parameters across ranks instead of replicating. Your likely depth project.

3D parallelism — DP + TP + PP combined. How frontier models actually train.

## Performance

Throughput ✓ — work per unit time. Tokens/sec for LLMs.

Latency — time for one operation. Your 30 tiny all-reduces are latency-bound; bucketing fixes that.

Bandwidth-bound vs latency-bound — is the cost the bytes moved, or the fixed per-call overhead? Determines which optimization helps.

Scaling efficiency — throughput(N) / (N × throughput(1)). Perfect linear scaling is 1.0. This is your headline metric at M3–M5.

Strong vs weak scaling — fixed total work split across more GPUs, vs. fixed work per GPU as you add more.

MFU (Model FLOPs Utilization) — achieved FLOPs ÷ hardware peak FLOPs. The standard "how well are you using the GPU" number; 40–55% is good for large training runs.

Overlap — running communication concurrently with computation so neither waits. M5.

Bucketing — batching many small tensors into one large buffer before communicating. M4.

Bubble — idle time where hardware waits. Pipeline parallelism's central problem.

Straggler — the slowest rank; collectives run at its pace, so one slow GPU throttles the whole job.

## GPU

Kernel — a function executed on the GPU. Your Triton project would write one.

Kernel launch overhead — fixed CPU cost per kernel call. Many tiny kernels = launch-bound. Why fusion helps.

Fusion — merging several ops into one kernel to avoid round-tripping through memory.

Memory-bound vs compute-bound — limited by moving bytes, or by arithmetic? Most elementwise ops are memory-bound; matmuls are compute-bound.

HBM vs SRAM — GPU main memory (large, slow) vs on-chip cache (tiny, fast). FlashAttention is fundamentally about keeping work in SRAM.

H2D / D2H — host-to-device and back, i.e. CPU↔GPU transfers.

Pinned memory — page-locked host memory that enables async DMA transfers. Project #6.

CUDA stream — an ordered queue of GPU work. Separate streams run concurrently, which is the mechanism behind overlap.

Async execution — CUDA calls return before the GPU finishes. Why you need torch.cuda.synchronize() before timing anything, or your measurements are fiction.

## Training internals

Optimizer state ✓ — what Adam keeps per parameter (momentum + variance). Typically 2× the model size in fp32, which is why sharding it matters.

Gradient accumulation — several forward/backward passes before one optimizer step, to simulate a larger batch than memory allows.

Activation checkpointing — discard activations during forward, recompute them during backward. Trades compute for memory.

Mixed precision / bf16 / AMP — compute in low precision, keep a master copy in fp32.

Global vs per-rank batch ✓ — the distinction that made your M2 comparison valid.

Determinism — same inputs → bit-identical outputs. The hard requirement in M6.

Checkpoint / resume — save enough state to restart exactly. M6.

Elastic training — surviving rank failures by re-running rendezvous with a new world size.