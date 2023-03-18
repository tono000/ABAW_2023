#!/bin/bash

trap "exit" INT
# shellcheck disable=SC2068

export WANDB_API_KEY='06f9860a9087485ea016ff31e51f8da0e96cb4b4'
export TORCH_HOME='/home/tu/ABAW3/Affwild2-ABAW3-ActionUnit/pretrained/'
export WANDB_CONFIG_DIR='/home/tu/ABAW3/Affwild2_ABAW5-main/'
export WANDB_CACHE_DIR='/home/tu/ABAW3/Affwild2_ABAW5-main/'


# Submission 01
test_config='/home/tu/ABAW3/config_VIVIT.yaml'   # Path to config file
test_ckpt='/home/tu/ABAW3/epoch=18-step=8626.ckpt'

python -W ignore main.py --cfg $test_config \
                          TEST_ONLY $test_ckpt

echo "Finished generate results for submission 01"
#
# # Submission 02
# test_config='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_14-45-39/config.yaml'   # Path to config file
# test_ckpt='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_14-45-39/version_None/checkpoints/epoch=19-step=6839.ckpt'
#
# python -W ignore main.py --cfg $test_config \
#                           TEST_ONLY $test_ckpt
#
# echo "Finished generate results for submission 02"
#
# # Submission 03
# test_config='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-23_12-47-38/config.yaml'   # Path to config file
# test_ckpt='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-23_12-47-38/version_None/checkpoints/epoch=19-step=6839.ckpt'
#
# python -W ignore main.py --cfg $test_config \
#                           TEST_ONLY $test_ckpt
#
# echo "Finished generate results for submission 03"
#
# # Submission 04
# test_config='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_16-44-13/config.yaml'   # Path to config file
# test_ckpt='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_16-44-13/version_None/checkpoints/epoch=17-step=6155.ckpt'
#
# python -W ignore main.py --cfg $test_config \
#                           TEST_ONLY $test_ckpt
#
# echo "Finished generate results for submission 04"
