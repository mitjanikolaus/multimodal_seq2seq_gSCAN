#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/test_learner.out

source activate gscan

#python -u -m teacher_learner_training --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --splits=test,visual_easier,visual,situational_1,situational_2,contextual,adverb_1,adverb_2 --max_decoding_steps=120 --max_testing_examples 2000 --test_checkpoints 192053 192100 192200 192300 192400 192500 192600 192700 192800 192900 193000 194000 195000 196000 197000 198000 199000 200000 210000 220000 230000 240000 250000 --test_dir out_learner_both_avg
python -u -m teacher_learner_training --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --splits=test,visual_easier,visual,situational_1,situational_2,contextual,adverb_1,adverb_2 --max_decoding_steps=120 --max_testing_examples 2000 --test_checkpoints 210000 220000 230000 240000 250000 --test_dir out_learner_both_avg

#python -u -m teacher_learner_training --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --splits=test,visual_easier,visual,situational_1,situational_2,contextual,adverb_1,adverb_2 --max_decoding_steps=120 --max_testing_examples 2000 --test_checkpoints 100 200 300 400 500 1000 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 150000 200000 250000 --test_dir out_learner_t_l_no_pretraining
