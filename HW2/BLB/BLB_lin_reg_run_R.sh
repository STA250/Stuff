#!/bin/bash -l

module load R

#SBATCH --job-name=blbfit
#SBATCH --mem-per-cpu=6000
#SARRAY --range=1-250

# Email notifications (optional), type=BEGIN, END, FAIL, ALL
##SBATCH --mail-type=ALL
##SBATCH --mail-user=pdbaines@ucdavis.edu

# Standard out and Standard Error output files with the job number in the name.
#SBATCH -o dump/BLB_lin_reg_job_%j.out
#SBATCH -e dump/BLB_lin_reg_job_%j.err

# Execute each of the jobs with a different index (the R script will then process
# this to do something different for each index):
srun R --vanilla --no-save --args ${SLURM_ARRAYID} < BLB_lin_reg_job.R 


