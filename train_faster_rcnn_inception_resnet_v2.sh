#!/bin/bash

#export PYTHONPATH=$PYTHONPATH

CONFIG_PATH='SceneText/faster_rcnn/faster_rcnn_inception_resnet_v2/faster_rcnn_inception_resnet_v2_atrous_text.config'
TRAIN_DIR='SceneText/faster_rcnn/faster_rcnn_inception_resnet_v2/train/'

CUDA_VISIBLE_DEVICES=1 python2 train.py \
    --logtostderr \
    --pipeline_config_path=${CONFIG_PATH} \
    --train_dir=$TRAIN_DIR
