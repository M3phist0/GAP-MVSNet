#!/usr/bin/env bash
TESTPATH="/root/gpufree-data/dtu" 						# path to dataset dtu_test
TESTLIST="lists/dtu/test.txt"							# path to data_list
NORMALPATH="dtu_normal"
CKPT_FILE="/root/gpufree-data/GAP-MVSNet/outputs/dtu_training/model_000010.ckpt"	    # path to checkpoint file
OUTDIR="./outputs/dtu_test" 						  # path to output
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


CUDA_VISIBLE_DEVICES=0 python test.py \
--dataset=general_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--normalpath=$NORMALPATH \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--numdepth=192 \
--num_view=5 \
--ndepths="48,32,8" \
--depth_inter_r="4.0,1.0,0.5" \
--interval_scale=1.06 \
--filter_method="o3d"
