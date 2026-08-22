import torch
import torch.distributed as dist


class GradientBucketer:
    def __init__(self, params, bucket_mb=25):
        self.bucket_bytes = int(bucket_mb * 1024 * 1024)
        self.params = list(params)
        self.buckets = self._build_buckets()

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

    def sync(self, world_size):
        for bucket in self.buckets:
            bucket_params = [p for p in bucket if p.grad is not None]
            if not bucket_params:
                continue

            grads = [p.grad for p in bucket_params]
            flat = torch._utils._flatten_dense_tensors(grads)

            if dist.is_initialized():
                dist.all_reduce(flat)
            flat.div_(world_size)

            unflat = torch._utils._unflatten_dense_tensors(flat, grads)
            for p, u in zip(bucket_params, unflat):
                p.grad.copy_(u)
