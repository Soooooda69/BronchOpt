#!/bin/bash

python test.py --dataroot datasets/colon_fullres \
	--name colon_cyclegan_noMI_oblique \
 	--model cycle_gan --epoch 30 \
 	--gpu_ids 0 \
	--phase train \
 	--num_test 24020 \
	--eval
