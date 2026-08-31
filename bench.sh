#!/usr/bin/env bash
set -e
BIG="--d-model 768 --n-layer 12 --n-head 12 --seq-len 512 --vocab-size 512 --dtype bf16"
STEPS="--steps 300 --log-every 50"

for N in 1 2 4; do
  torchrun --nproc_per_node=$N train_ddp.py $BIG $STEPS --batch-size 32 --run-name "scale_${N}rank"
done

for MB in 1 5 25 100; do
  torchrun --nproc_per_node=4 train_ddp.py $BIG $STEPS --batch-size 32 --bucket-mb $MB --run-name "bucket_${MB}mb"
done

torchrun --nproc_per_node=4 train_ddp.py \
  --d-model 768 --n-layer 12 --n-head 12 --seq-len 512 --vocab-size 512 --dtype fp32 \
  $STEPS --batch-size 32 --run-name "fp32_4rank"

torchrun --nproc_per_node=4 train_ddp.py $BIG --steps 30 --log-every 10 \
  --batch-size 32 --profile --profile-steps 5 --run-name "prof_4rank"
