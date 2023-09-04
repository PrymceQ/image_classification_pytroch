#!/bin/bash

# 设置可见的CUDA设备
export CUDA_VISIBLE_DEVICES=0,1  # GPU 0,1

# 设置训练参数
CFG=config/default_config.yaml
SAVED_PATH='weights/example/'

# 执行训练命令
python train.py --cfg $CFG --saved_path $SAVED_PATH
