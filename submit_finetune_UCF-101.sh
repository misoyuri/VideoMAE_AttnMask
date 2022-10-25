#!/bin/sh

# >>> Job name <<< #
#SBATCH -J C8G2_KSU_VideoMAE_1017_finetune

# >>> Partition name (queue) <<< #
#SBATCH -p pleiades

# >>> Per node <<< #
#SBATCH --nodes=1

# >>> Core per node <<< #
#SBATCH --ntasks-per-node=1

# >>> Number of GPUs <<< #
#SBATCH --gres=gpu:4

# >>> Output <<< #
#SBATCH -o %x.%j.o

# >>> Error <<< #
#SBATCH -e %x.%j.e

OUTPUT_DIR='./output/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800/eval_lr_5e-4_epoch_50'
DATA_PATH='/home/Data/UCF-101/Labels'
MODEL_PATH='/home/s20225004/Video/VideoMAE/outputs/UCF-101_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800/checkpoint-799.pth'

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