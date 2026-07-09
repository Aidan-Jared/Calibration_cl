#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed 42\
  --model-runs 5\
  --lr 3e-2\
  --momentum 0.9\
  --batch-size 32\
  --task-epochs 10\
  --transform "True"\
  --dropout 0.0\
  --data_set "CIFAR10"\
  --task-splits 5\
  --model "singleHeadResNet32"\
  --norm "[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]"\
  --method "DER"\
  --der-alpha .1\
  --der-beta 0.5\
  --buffer-size 200\
  --replay-size 32\
  --task-shuffle "False"
