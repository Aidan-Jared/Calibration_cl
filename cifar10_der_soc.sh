#!/bin/bash

# Exit on error
set -e

uv run main.py \
  --seed "[42]"\
  --lr 3e-3\
  --batch-size 32\
  --task-epochs 10\
  --dropout 0.1\
  --data_set "CIFAR10"\
  --task-splits 5\
  --model "singleHeadResNet32"\
  --norm "[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]"\
  --method "DER"\
  --der-alpha .1\
  --der-beta .5\
  --buffer-size 500\
  --loss "socrates"\
  --soc-alpha 0.5\
  --soc-gamma 1.0
