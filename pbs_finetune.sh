#!/bin/bash 

#PBS -l select=1:ncpus=12:ngpus=4:mem=50GB
#PBS -q pleiades1
#PBS -r n 
#PBS -N G4C8_KSU_VM_finetune_1024
#PBS -o G4C8_KSU_VM_finetune_1024.o
#PBS -e G4C8_KSU_VM_finetune_1024.e

source activate DAT

cd $PBS_O_WORKDIR
DATE=`date +%y%m%d`
echo $PBS_JOBID


OUTPUT_DIR=outputs/221022/UCF-101_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800/eval_lr_5e-4_epoch_50
DATA_PATH='/home/Data/UCF-101/Labels'
MODEL_PATH=/home/s20225004/Video/VideoMAE_MILAN/outputs/221022/UCF-101_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800/checkpoint-799.pth

torchrun --nproc_per_node=4 run_class_finetuning_modified.py \
    --model vit_base_patch16_224 \
    --data_set UCF101 \
    --nb_classes 101 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 6 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 