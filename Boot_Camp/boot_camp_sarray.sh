#!/bin/bash -l
# NOTE the -l flag!

###############################################################################
##
## NOTES:
## 
## (1) When specifying --range as a range it must start from a positive
##     integer e.g.,
##       SARRAY --range=0-9 
##     is not allowed.
##
## (2) Negative numbers are not allowed in --range
##     e.g.,
##      SARRAY --range=-5,-4,-3,-2,-1,0,1,2,3,4,5
##     is not allowed.
##
## (3) Zero can be included if specified separately.
##    e.g., 
##       SARRAY --range=0,1-9
##     is allowed.
##
## (4) Ranges can be combined with specified job numbers.
##    e.g., 
##       SARRAY --range=0,1-4,6-10,50-100
##     is allowed.
##
###############################################################################

# Name of the job - You'll probably want to customize this.
#SBATCH --job-name=boot_camp_demo
#SARRAY --range=0,1-19

# Standard out and Standard Error output files with the job number in the name.
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err

# Execute each of the jobs with a different index (the python script will then process
# this to do something different for each index):
srun python boot_camp_demo.py -o out_${SLURM_ARRAYID}.txt 


