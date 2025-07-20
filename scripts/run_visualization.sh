#!/bin/bash

model_ckpt='./models_ckpt/ImagenFew/dyConv_Basic_24.ckpt'
dataset=ETTh1

python run_visualization.py --model_ckpt $model_ckpt --config configs/finetune/$dataset.yaml