#!/bin/bash -l

module load R/3.0.0

#SBATCH --job-name=blr_post_process
#SBATCH --output=dump/blr_post_process.out
#SBATCH --error=dump/blr_post_process.err

srun R --no-save --vanilla < BLR_post_process.R



