#!/usr/bin/env bash

#TESTPATH="data/tankandtemples/intermediate" # path to dataset
#TESTLIST="lists/tnt/inter.txt"
# NORMAL_PATH="MoGe/tnt_normal/intermediate" 			

TESTPATH="data/tankandtemples/advanced" # path to dataset
TESTLIST="lists/tnt/adv.txt"
NORMAL_PATH="MoGe/tnt_normal/advanced" 	
											
CKPT_FILE="outputs/bld_finetune/model_000015.ckpt" 		    # path to checkpoint
OUTDIR="outputs/tnt_testing" 									# path to save the results

if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi

CUDA_VISIBLE_DEVICES=0 python test.py \
--max_h 960 \
--max_w 1920 \
--dataset=tnt_eval \
--num_view=11 \
--batch_size=1 \
--normalpath=$NORMAL_PATH \
--interval_scale=1.0 \
--numdepth=192 \
--ndepths="48,32,8"  \
--depth_inter_r="4,1,0.25" \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--outdir=$OUTDIR  \
--filter_method="dynamic" \
--loadckpt $CKPT_FILE ${@:2}

#Using this script to generate depth maps and then run the dynamic_fusion.sh to generate the final point cloud.

