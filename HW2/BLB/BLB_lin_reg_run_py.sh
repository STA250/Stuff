#!/bin/bash -l

module load pymods/2.7
module load numpy

#SBATCH --job-name=blbfit
#SBATCH --mem-per-cpu=6000
#SARRAY --range=1-250

# Email notifications (optional), type=BEGIN, END, FAIL, ALL
##SBATCH --mail-type=ALL
##SBATCH --mail-user=pdbaines@ucdavis.edu

# Standard out and Standard Error output files with the job number in the name.
#SBATCH -o dump/BLB_lin_reg_job_%j.out
#SBATCH -e dump/BLB_lin_reg_job_%j.err

# Execute each of the jobs with a different index
# (the python script will then process
# this to do something different for each index):
srun python BLB_lin_reg_job.py -i ${SLURM_ARRAYID}

