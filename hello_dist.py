"""Smallest possible torchrun sanity check: does collective communication work?

    torchrun --nproc_per_node=4 hello_dist.py

Each rank builds a one-element tensor holding its own rank and all-reduces it.
With SUM, every rank should end up with 0+1+...+(world_size-1).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main():

    dist.init_process_group(backend="gloo")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])


    dist_rank = dist.get_rank()
    dist_world_size = dist.get_world_size()

    assert rank == dist_rank, f"env RANK {rank} != get_rank() {dist_rank}"
    assert world_size == dist_world_size, f"env WORLD_SIZE {world_size} != get_world_size() {dist_world_size}"

    t = torch.tensor([float(rank)])
    dist.all_reduce(t)  # op defaults to ReduceOp.SUM

    expected = world_size * (world_size - 1) / 2

    msg = (
        f"[rank {rank}] local_rank={local_rank} world_size={world_size} "
        f"all_reduce={t.item()} expected={expected}\n"
    )
    
    # non-deterministic print order: all ranks print at once, which can interleave and be confusing
    # print(msg, end="", flush=True)

    # deterministic print order: each rank prints in turn, with a barrier between each print
    for r in range(world_size):
        if r == rank:
            print(msg, end="", flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
