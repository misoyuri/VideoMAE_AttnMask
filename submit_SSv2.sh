#!/bin/sh

# >>> Job name <<< #
#SBATCH -J C8G2_KSU_VideoMAE_1015

# >>> Partition name (queue) <<< #
#SBATCH -p pleiades

# >>> Per node <<< #
#SBATCH --nodes=1

# >>> Core per node <<< #
#SBATCH --ntasks-per-node=1

# >>> Number of GPUs <<< #
#SBATCH --gres=gpu:2

# >>> Output <<< #
#SBATCH -o %x.%j.o

# >>> Error <<< #
#SBATCH -e %x.%j.e

OUTPUT_DIR='./ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800'
DATA_PATH='YOUR_PATH/list_ssv2/train.csv'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12320 \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}

        