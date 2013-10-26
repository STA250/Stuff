#!/bin/bash -l

###############################################################################
##
## NOTES:
## 
## (1) When specifying --range as a range it must start from a positive
##     integer e.g.,
##       #SARRAY --range=0-9 
##     is not allowed.
##
## (2) Negative numbers are not allowed in --range
##     e.g.,
##    %   #SARRAY --range=-5,-4,-3,-2,-1,0,1,2,3,4,5
##     is not allowed.
##
## (3) Zero can be included if specified separately.
##    e.g., 
##       #SARRAY --range=0,1-9
##     is allowed.
##
## (4) Ranges can be combined with specified job numbers.
##    e.g., 
##       #SARRAY --range=0,1-4,6-10,50-100,1001-1002
##     is allowed.
##
###############################################################################

module load R/3.0.0

# Name of the job - You'll probably want to customize this.
#SBATCH --job-name=blr_sim_data
# Specify range of jobs to run - passed into R as 'args'
#SARRAY --range=1-200

# Standard out and Standard Error output files with the job number in the name.
#SBATCH --output=dump/blr_sim_data_%j.out
#SBATCH --error=dump/blr_sim_data_%j.err

# Execute each of the jobs with a different index (the R script will then process
# this to do something different for each index):
srun R --no-save --vanilla --args ${SLURM_ARRAYID} < BLR_sim.R



