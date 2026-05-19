#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed 42\
  --model-runs 5\
  --lr .03\
  --batch-size 32\
  --task-epochs 5\
  --dropout 0.1\
  --data_set "CIFAR100"\
  --task-splits 10\
  --model "singleHeadResNet32"\
  --norm "[(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)]"\
  --method "DER"\
  --der-alpha .2\
  --der-beta .5\
  --buffer-size 1000\
  --replay-size 32\
  --task-shuffle "True"
