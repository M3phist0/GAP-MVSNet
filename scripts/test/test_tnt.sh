#!/usr/bin/env bash
#TESTPATH="path/to/TanksAndTemples/intermediate" # path to dataset
TESTPATH="path/to/TanksAndTemples/advanced" # path to dataset
#TESTLIST="lists/tnt/inter.txt"
TESTLIST="lists/tnt/adv.txt"
NORMAL_PATH="path/to/tnt_normal/" 												
CKPT_FILE="path/to/ckpt" 		    # path to checkpoint
OUTDIR="outputs/tnt_testing/" 									# path to save the results
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi

CUDA_VISIBLE_DEVICES=6 python test.py \
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

