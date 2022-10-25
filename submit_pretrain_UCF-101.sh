#!/bin/sh

# >>> Job name <<< #
#SBATCH -J C1G1_KSU_Fusion

# >>> Partition name (queue) <<< #
#SBATCH -p pleiades

# >>> Per node <<< #
#SBATCH --nodes=1

# >>> Core per node <<< #
#SBATCH --ntasks-per-node=1

# >>> Number of GPUs <<< #
#SBATCH --gres=gpu:1

# >>> Output <<< #
#SBATCH -o %x.%j.o

# >>> Error <<< #
#SBATCH -e %x.%j.e

OUTPUT_DIR='outputs/UCF-101_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800'
DATA_PATH='/home/Data/UCF-101/Labels/train.csv'

torchrun --master_port 14595 --nproc_per_node=1 \
        --node_rank=0 \
        run_mae_pretraining_modified.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 24 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}

