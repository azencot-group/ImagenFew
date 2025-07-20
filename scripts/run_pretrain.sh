#!/bin/bash

dataset=pretrain

python run.py --no_test_model --seq_len 24 --ddp --config configs/pretrain/$dataset.yaml