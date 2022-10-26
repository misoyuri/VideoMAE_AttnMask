#!/bin/bash 

#PBS -l select=1:ncpus=8:ngpus=2:mem=50GB
#PBS -q pleiades1
#PBS -r n 
#PBS -N G2C8_KSU_1025_UCF101_videomae_pretrain_base_patch16_224_frame_16x4_attn_mask_ratio_0.9_e800
#PBS -o G2C8_KSU_1025_UCF101_videomae_pretrain_base_patch16_224_frame_16x4_attn_mask_ratio_0.9_e800.o
#PBS -e G2C8_KSU_1025_UCF101_videomae_pretrain_base_patch16_224_frame_16x4_attn_mask_ratio_0.9_e800.e

source activate DAT

cd $PBS_O_WORKDIR
DATE=`date +%y%m%d`
echo $PBS_JOBID


OUTPUT_DIR=outputs/$DATE/UCF-101_videomae_pretrain_base_patch16_224_frame_16x4_attn_mask_ratio_0.9_e800
DATA_PATH=/home/Data/UCF-101/Labels/train.csv

torchrun --master_port 14595 --nproc_per_node=2 \
        --node_rank=0 \
        run_mae_pretraining_modified.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --tubelet_size 2 \
        --decoder_embedding_dim 1536 \
        --batch_size 24 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} > G2C8_KSU_1025_UCF101_videomae_pretrain_base_patch16_224_frame_16x4_attn_mask_ratio_0.9_e800.txt
