#!/bin/bash
#
#SBATCH --job-name=t_l
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/train_learner_t_l.out

source activate gscan

# train model on teacher train split
python -u -m teacher_learner_training --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/teacher_small_50 --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=out_learner_t_l --training_batch_size=200 --max_training_iterations=250000 --seed=1 --resume_from_file_teacher out_teacher_lm_small_50/model_best.pth.tar --resume_from_file_learner out_pre_train_learner/model_best.pth.tar --weight_lm_loss 1 --evaluate_every 100 --objective teacher-to-learner


# --reset_optimizer --learning_rate 0.0001

# check: lm loss? reset optimizer? learning_rate?

#For debugging enable: --max_training_examples 100


