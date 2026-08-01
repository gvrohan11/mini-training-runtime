#!/usr/bin/env bash
# usage: ./run_dist.sh <nproc> <script> [args...]
NPROC=$1; shift
OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo0 USE_LIBUV=0 \
python -m torch.distributed.run --nproc_per_node="$NPROC" \
  --master_addr=127.0.0.1 --master_port=29501 "$@"
