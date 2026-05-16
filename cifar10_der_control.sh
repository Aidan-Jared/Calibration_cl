#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed 42\
  --model-runs 1\
  --lr 1e-3\
  --batch-size 32\
  --task-epochs 10\
  --dropout 0.0\
  --data_set "CIFAR10"\
  --task-splits 5\
  --model "singleHeadResNet32"\
  --norm "[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]"\
  --method "DER"\
  --der-alpha .1\
  --der-beta .75\
  --buffer-size 1000\
  --replay-size 64\
  --task-shuffle "True"
