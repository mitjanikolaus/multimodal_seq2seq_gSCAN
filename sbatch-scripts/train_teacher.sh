#!/bin/bash
#
#SBATCH --job-name=gscan
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train.out
#SBATCH --error=out/train.out

source activate gscan

# train model on teacher train split
python -u -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/teacher --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=out_teacher --training_batch_size=200 --max_training_iterations=200000 --seed=1

