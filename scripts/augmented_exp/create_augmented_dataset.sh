#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=create_augmented_dataset

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=qiwei.peng@sussex.ac.uk

# save the log to the specified location
#SBATCH --output=./scripts/augmented_exp/create_augmented_dataset.log

# run the application
module purge
module load cuda/11.2
module load python/anaconda3
source $condaDotFile
source activate root
conda activate just_rank_augment

#export WANDB_PROJECT=valentwin_dstl
#export WANDB_MODE=offline

python src/augment_dataset.py