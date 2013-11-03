#!/bin/bash -l

module load R

#SBATCH --job-name=blb_post_process
#SBATCH --output=dump/blb_post_process.out
#SBATCH --error=dump/blb_post_process.err

srun R --no-save --vanilla < BLB_lin_reg_process.R



