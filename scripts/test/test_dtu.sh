#!/usr/bin/env bash
TESTPATH="/mnt/sharedisk/chenkehua/DTU/dtu_eval" 						# path to dataset dtu_test
TESTLIST="lists/dtu/test.txt"							# path to data_list
NORMALPATH="normalprior"
CKPT_FILE="paht/to/ckpt"	    # path to checkpoint file
OUTDIR="./outputs/dtu_test" 						  # path to output
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


CUDA_VISIBLE_DEVICES=7 python test.py \
--dataset=general_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--normalpath=$NORMALPATH \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--numdepth=192 \
--num_view=3 \
--ndepths="48,32,8" \
--depth_inter_r="4.0,1.0,0.5" \
--interval_scale=1.06 \
--filter_method="o3d"
