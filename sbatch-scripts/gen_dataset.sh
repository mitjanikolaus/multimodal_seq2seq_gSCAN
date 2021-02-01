#!/bin/bash
#
#SBATCH --job-name=gscan
#
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=32000
#SBATCH --output=out/gen_dataset.out
#SBATCH --error=out/gen_dataset.out

source activate gscan

# generate teacher dataset
python -m GroundedScan --mode generate --output_directory "train_teacher" --save_dataset_as "dataset.txt" --make_dev_set --split generalization --max_examples 2000000 --exclude_samples data/compositional_splits/dataset.txt



