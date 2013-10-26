
##
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##

library(mvtnorm)
library(coda)

########################################################################################
########################################################################################
## Handle batch job arguments:

# 1-indexed version is used now.
args <- commandArgs(TRUE)

cat(paste0("Command-line arguments:\n"))
print(args)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start <- 1000
length.datasets <- 200
#######################

if (length(args)==0){
  sinkit <- FALSE
  sim_num <- sim_start + 1
  set.seed(1330931)
} else {
  # Sink output to file?
  sinkit <- TRUE
  # Decide on the job number, usually start at 1000:
  sim_num <- sim_start + as.numeric(args[1])
  # Set a different random seed for every job number!!!
  set.seed(762*sim_num + 1330931)
}

# Simulation datasets numbered 1001-1200

########################################################################################
########################################################################################



"bayes.logreg" <- function(n,y,X,beta.0,Sigma.0.inv,niter=10000,burnin=1000,
                           print.every=1000,retune=100,verbose=TRUE)
{
	# Stuff
}

#################################################
# Set up the specifications:
beta.0 <- matrix(c(0,0))
Sigma.0.inv <- diag(rep(1.0,p))
niter <- 10000
# etc... (more needed here)
#################################################

# Read data corresponding to appropriate sim_num:

# Extract X and y:

# Fit the Bayesian model:

# Extract posterior quantiles...

# Write results to a (99 x p) csv file...

# Go celebrate.
 
cat("done. :)\n")







