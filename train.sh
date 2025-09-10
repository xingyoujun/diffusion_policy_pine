#!/bin/bash
sleep 5s
echo sleep 5s
sleep 1h
CUDA_VISIBLE_DEVICES=5 python train.py --config-name=train_diffusion_unet_real_image_workspace task=real_pine_stackcup_i4d training.num_epochs=601
sleep 2h
CUDA_VISIBLE_DEVICES=5 python train.py --config-name=train_diffusion_unet_real_image_workspace task=real_pine_stackcup_i4d training.num_epochs=601
sleep 2h
CUDA_VISIBLE_DEVICES=5 python train.py --config-name=train_diffusion_unet_real_image_workspace task=real_pine_stackcup_i4d training.num_epochs=601
sleep 2h
CUDA_VISIBLE_DEVICES=5 python train.py --config-name=train_diffusion_unet_real_image_workspace task=real_pine_stackcup_i4d training.num_epochs=601