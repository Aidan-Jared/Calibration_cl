#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed "[42]"\
  --lr 1e-3\
  --batch_size 16\
  --task_epochs 25\
  --dropout 0.1\
  --data_set "CIFAR10"\
  --task_splits 5\
  --model "singleHeadResNet32"\
  --norm "[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]"\
  --method "DER"\
  --der-alpha .5\
  --der-beta .5\
  --buffer_size 1000\