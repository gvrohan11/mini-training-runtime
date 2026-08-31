import torch
import torch.distributed as dist


class GradientBucketer:
    def __init__(self, params, bucket_mb=25, comm_dtype=None):
        self.bucket_bytes = int(bucket_mb * 1024 * 1024)
        self.params = list(params)
        self.comm_dtype = comm_dtype
        self.buckets = self._build_buckets()
        self.pending = [len(bucket) for bucket in self.buckets]
        self.inflight = []

    def _build_buckets(self):
        buckets = []
        cur = []
        cur_bytes = 0

        for p in reversed(self.params):
            size = p.numel() * p.element_size()
            if cur and cur_bytes + size > self.bucket_bytes:
                buckets.append(cur)
                cur = []
                cur_bytes = 0
            cur.append(p)
            cur_bytes += size

        if cur:
            buckets.append(cur)

        return buckets

    def register_hooks(self):
        for bucket_idx, bucket in enumerate(self.buckets):
            for p in bucket:
                p.register_post_accumulate_grad_hook(
                    lambda param, _idx=bucket_idx: self._mark_bucket_ready(_idx)
                )

    def _mark_bucket_ready(self, bucket_idx):
        self.pending[bucket_idx] -= 1
        if self.pending[bucket_idx] != 0:
            return

        bucket = self.buckets[bucket_idx]
        grads = [p.grad for p in bucket]

        flat = torch._utils._flatten_dense_tensors(grads)
        if self.comm_dtype is not None:
            flat = flat.to(self.comm_dtype)
        print(f"[bucket {bucket_idx}] flat.dtype = {flat.dtype}")  # temporary
        handle = dist.all_reduce(flat, async_op=True)
        self.inflight.append((handle, flat, grads))

    def wait_and_copy(self, world_size):
        for handle, flat, grads in self.inflight:
            handle.wait()

        for handle, flat, grads in self.inflight:
            flat.div_(world_size)
            unflat = torch._utils._unflatten_dense_tensors(flat, grads)
            for g, u in zip(grads, unflat):
                g.copy_(u)

        self.inflight.clear()
        self.pending = [len(bucket) for bucket in self.buckets]

    def sync(self, world_size):
        for bucket in self.buckets:
            bucket_params = [p for p in bucket if p.grad is not None]
            if not bucket_params:
                continue

            grads = [p.grad for p in bucket_params]
            flat = torch._utils._flatten_dense_tensors(grads)
            if self.comm_dtype is not None:
                flat = flat.to(self.comm_dtype)

            if dist.is_initialized():
                dist.all_reduce(flat)
            flat.div_(world_size)

            unflat = torch._utils._unflatten_dense_tensors(flat, grads)
            for p, u in zip(bucket_params, unflat):
                p.grad.copy_(u)
