import torch
import torch.distributed as dist


def partition_params(params, world_size):
    """Partition a parameter list into a deterministic shard layout.

    The split is greedy by parameter element count so large tensors are spread
    roughly evenly across ranks. This keeps the per-rank optimizer state balanced
    without depending on rank-local state.
    """
    params = list(params)
    if world_size <= 1:
        return [params]

    shards = [[] for _ in range(world_size)]
    shard_numel = [0 for _ in range(world_size)]

    for p in params:
        owner = min(range(world_size), key=lambda i: (shard_numel[i], i))
        shards[owner].append(p)
        shard_numel[owner] += p.numel()

    return shards


class ZeroOptimizer:
    """ZeRO-1 optimizer wrapper.

    - Every rank keeps optimizer state only for its own shard.
    - Gradients are still reduced across ranks as normal.
    - After the local AdamW step, each rank re-broadcasts updated parameters from
      their owner so all ranks end the step with identical model weights.
    """

    def __init__(self, params, optimizer_factory, rank=None, world_size=None):
        self.params = list(params)
        if world_size is None:
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        else:
            self.world_size = int(world_size)

        if rank is None:
            self.rank = dist.get_rank() if dist.is_initialized() else 0
        else:
            self.rank = int(rank)

        self.shards = partition_params(self.params, self.world_size)
        self.local_params = self.shards[self.rank]
        self.param_to_owner = {}
        for owner, shard in enumerate(self.shards):
            for p in shard:
                self.param_to_owner[id(p)] = owner

        self.optimizer = optimizer_factory(self.local_params)

    def __getattr__(self, name):
        # Note: this exposes only the local optimizer shard, not the full parameter
        # set. Any scheduler or wrapper that expects all model parameters via
        # param_groups will silently operate on a subset of the model.
        return getattr(self.optimizer, name)

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.params:
            if p.grad is not None:
                p.grad = None if set_to_none else p.grad.zero_()

    def _sync_params(self):
        if not dist.is_initialized():
            return

        for p in self.params:
            owner = self.param_to_owner[id(p)]
            dist.broadcast(p.data, src=owner)

    def step(self, closure=None):
        if closure is None:
            ret = self.optimizer.step()
        else:
            ret = self.optimizer.step(closure)
        self._sync_params()
        return ret

    def state_dict(self):
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "optimizer" in state_dict:
            state_dict = state_dict["optimizer"]
        return self.optimizer.load_state_dict(state_dict)
