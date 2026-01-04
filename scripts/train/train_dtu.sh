#!/usr/bin/env bash
MVS_TRAINING="/mnt/sharedisk/chenkehua/DTU/mvs_training/dtu" # path to dataset mvs_training
NORMAL_PATH="/mnt/sharedisk/chenkehua/networks/Lotus/output/dtutrain"
#NORMAL_PATH="/mnt/sharedisk/chenkehua/GS+Nerf/CityGaussian/utils/da2_train"
#NORMAL_PATH="/mnt/sharedisk/chenkehua/networks/GoMVS/GT_Normal"
LOG_DIR="./outputs/dtu_training" # path to checkpoints
if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi

NGPUS=4
BATCH_SIZE=1
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=8888 train.py \
	--logdir=$LOG_DIR \
	--dataset=dtu_yao \
	--batch_size=$BATCH_SIZE \
	--epochs=16 \
	--trainpath=$MVS_TRAINING \
	--normalpath=$NORMAL_PATH \
	--trainlist=lists/dtu/train.txt \
	--testlist=lists/dtu/val.txt \
	--numdepth=192 \
	--ndepths="48,32,8" \
	--nviews=5 \
	--wd=0.0001 \
	--mode="train" \
	--depth_inter_r="4.0,1.0,0.5" \
	--lrepochs="6,8,12:2" \
	--dlossw="1.0,1.0,1.0" | tee -a $LOG_DIR/log.txt # > traindtu.log 2>&1 &

