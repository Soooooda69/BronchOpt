#!/bin/bash

# python ./third_party/sp_cyclegan/train.py \
# 	--continue_train \
# 	--netG unet_128 \
# 	--dataroot ./processed/train_cyclegan_rgb_depth \
# 	--name rgb_struct_cyclegan \
# 	--model cycle_gan_struct \
# 	--n_epochs 300 \
# 	--lambda_struct 1.0 \
# 	--gpu_ids 0 \
# 	--batch_size 6 \
# 	--display_id -1

# python ./third_party/sp_cyclegan/train.py \
# 	--dataroot ./processed/train_cyclegan_rgb_depth \
# 	--name rgb_struct_cyclegan_res9 \
# 	--model cycle_gan_struct \
# 	--n_epochs 300 \
# 	--lambda_struct 1.0 \
# 	--gpu_ids 0 \
# 	--batch_size 6 \
# 	--display_id -1

python ./third_party/sp_cyclegan/train.py \
	--dataroot ./processed/train_cyclegan_rgb_depth \
	--netG unet_256 \
	--name rgb_struct_cyclegan_u256 \
	--model cycle_gan_struct \
	--n_epochs 300 \
	--lambda_struct 1.0 \
	--gpu_ids 0 \
	--batch_size 6 \
	--display_id -1