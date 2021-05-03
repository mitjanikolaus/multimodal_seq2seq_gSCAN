#!/bin/bash
#
#SBATCH --job-name=gscan
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/train_teacher_lm_50_pointing.out

source activate gscan

# train model on teacher train split
python -u -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/teacher_small_50 --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=out_teacher_lm_small_50_pointing --training_batch_size=200 --max_training_iterations=200000 --seed=1

