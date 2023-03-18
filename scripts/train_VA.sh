#!/bin/bash

trap "exit" INT
# shellcheck disable=SC2068

export WANDB_API_KEY='28f726b46ab7c4e23e7db4aefcc65e3fbaa06106'
export TORCH_HOME='/mnt/Work/Dataset/Affwild2_ABAW5/pretrained/'
export WANDB_CONFIG_DIR='/mnt/Work/Dataset/Affwild2_ABAW5/'
export WANDB_CACHE_DIR='/mnt/Work/Dataset/Affwild2_ABAW5/'

run_ver='v2'
task='VA'
logger='none' #'wandb' none

# Temporal configs
num_enc_dnc=2
tranf_nhead=5
tranf_dim_fc=512
seq_len=256 # old 256

# Backbone configs
model_backbone='regnet-400mf'
freeze_bn=True
model_aux=1.
fusion_strategy=1

# Optimizer configs
optim_name='adam'
lr_policy='cos-restart'  # 'cos-restart'
warmup_epoch=4
warmup_factor=0.1
wd=5e-5
max_epoch=20 # 25 # old 20
train_bsz=16 # old 16
test_bsz=16

base_lr=0.0005  # 0.9 # 0.0001 0.005

img_size=112
pretrained_model='none'  # 'none' #
train_dir='/mnt/Work/Dataset/Affwild2_ABAW5/train_logs/'$task'_'$run_ver'/'

test_only='none' #'/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-20_07-34-47/version_None/checkpoints/epoch=23-step=8207.ckpt'
# Run command
python -W ignore main.py --cfg conf/${task}_baseline.yaml \
        TASK $task \
        LOGGER $logger \
        OUT_DIR $train_dir \
        OPTIM.MAX_EPOCH $max_epoch \
        OPTIM.WARMUP_FACTOR $warmup_factor \
        OPTIM.BASE_LR $base_lr \
        OPTIM.NAME $optim_name \
        OPTIM.LR_POLICY $lr_policy \
        OPTIM.WEIGHT_DECAY $wd \
        OPTIM.WARMUP_EPOCHS $warmup_epoch \
        TRAIN.BATCH_SIZE $train_bsz \
        TEST.BATCH_SIZE $test_bsz \
        TRANF.NUM_ENC_DEC $num_enc_dnc \
        TRANF.NHEAD $tranf_nhead \
        TRANF.DIM_FC $tranf_dim_fc \
        MODEL.BACKBONE $model_backbone \
        MODEL.BACKBONE_FREEZE "'block4', 'block3', 'block2'" \
        MODEL.FREEZE_BATCHNORM $freeze_bn \
        MODEL.BACKBONE_PRETRAINED $pretrained_model \
        MODEL.USE_AUX $model_aux \
        MODEL.FUSION_STRATEGY $fusion_strategy \
        DATA_LOADER.SEQ_LEN $seq_len \
        DATA_LOADER.IMG_SIZE $img_size \
        TEST_ONLY $test_only \
        OPTIM.TUNE_LR False