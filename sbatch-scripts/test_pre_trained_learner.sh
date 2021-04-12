#!/bin/bash
#
#SBATCH --job-name=gscan
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --output=out/test_pre_train_learner.out

source activate gscan

#python -u -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=out_pre_train_learner --resume_from_file=out_pre_train_learner/model_best.pth.tar --splits=test,visual_easier,visual,situational_1,situational_2,contextual,adverb_1,adverb_2 --max_decoding_steps=120 --max_testing_examples 2000 &> out_pre_train_learner/test.txt

python -u -m teacher_learner_training --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=out_pre_train_learner --splits=test,visual_easier,visual,situational_1,situational_2,contextual,adverb_1,adverb_2 --max_decoding_steps=120 --max_testing_examples 2000 --test_checkpoints out_pre_train_learner/model_best.pth.tar

