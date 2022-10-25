#!/bin/bash 

#PBS -l select=1:ncpus=8:ngpus=1:mem=50GB
#PBS -q pleiades1
#PBS -r n 
#PBS -N G2C8_KSU_VM_Pretrain_1022
#PBS -o G2C8_KSU_VM_Pretrain_1022.o
#PBS -e G2C8_KSU_VM_Pretrain_1022.e

source activate DAT

cd $PBS_O_WORKDIR
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
        --decoder_embedding_dim 768 \
        --batch_size 8 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} 
