#!/bin/bash

model_ckpt='./models_ckpt/ImagenFew/dyConv_Basic_24.ckpt'
subset_n=50

for dataset in "Mujoco" "ETTh2" "ETTm1" "ETTm2" "Sine" "Weather" "ILI" "SaugeenRiverFlow" "ECG200" "SelfRegulationSCP1" "StarLightCurves" "AirQuality" 
do
  python run.py --subset_n $subset_n --model_ckpt $model_ckpt --config configs/finetune/$dataset.yaml
done