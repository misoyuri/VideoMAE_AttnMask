#!/bin/bash 

#PBS -l select=1:ncpus=12:ngpus=1
#PBS -q pleiades2
#PBS -r n 
#PBS -N G1C12_KSU_
#PBS -j oe

source activate DAT

cd $PBS_O_WORKDIR
mkdir /scratch/user

cp /home/user/submit /scratch/s20225004/
DATE=`date +%y%m%d`
echo $PBS_JOBID


OUTPUT_DIR=outputs/$DATE/UCF-101_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800
DATA_PATH=/home/Data/UCF-101/Labels/train.csv

torchrun --master_port 14595 --nproc_per_node=1 \
        --node_rank=0 \
        run_mae_pretraining_modified.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --tubelet_size 1 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        > 1025_batch32.txt

cp /scratch/user/xxxx /home/user/
rm -rf /scratch/user/xxxx

