#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed 42\
  --model-runs 5\
  --lr 3e-3\
  --momentum 0.5\
  --batch-size 32\
  --task-epochs 3\
  --dropout 0.1\
  --data_set "CIFAR10"\
  --task-splits 5\
  --model "singleHeadResNet18"\
  --norm "[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]"\
  --method "DER"\
  --der-alpha .5\
  --der-beta .75\
  --buffer-size 600\
  --replay-size 64\
  --task-shuffle "True"
