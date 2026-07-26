#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed 42\
  --model-runs 5\
  --lr 3e-2\
  --batch-size 32\
  --task-epochs 50\
  --transform "True"\
  --dropout 0.0\
  --data_set "CIFAR10"\
  --task-splits 5\
  --model "singleHeadResNet32"\
  --norm "[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]"\
  --method "DER"\
  --der-alpha .5\
  --der-beta .75\
  --buffer-size 200\
  --replay-size 32\
  --task-shuffle "False"\
  --loss "socrates"\
  --soc-alpha 0.5\
  --soc-gamma 1.0\
  --M 10
