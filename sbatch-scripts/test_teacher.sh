#!/bin/bash
#
#SBATCH --job-name=gscan
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/test.out
#SBATCH --error=out/test.out

source activate gscan


# test teacher model on test splits
#python -u -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=out_teacher --resume_from_file=out_teacher/model_best.pth.tar --splits=test,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=teacher.json --max_decoding_steps=120 &> out_teacher/test.txt

# test small teacher model
python -u -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=out_teacher_small_50 --resume_from_file=out_teacher_small_50/model_best.pth.tar --splits=test,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=teacher_small_50.json --max_decoding_steps=120 &> out_teacher_small_50/test.txt

#python -u -m seq2seq --mode=test --data_directory=data/test_variance --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=out_teacher_test_variance --resume_from_file=out_teacher_small/model_best.pth.tar --splits=adverb_1_split_1 --output_file_name=teacher_small.json --max_decoding_steps=120 &> out_teacher_test_variance/test.txt

