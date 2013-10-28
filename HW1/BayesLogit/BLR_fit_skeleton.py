
##
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##

import sys
import numpy as np

########################################################################################
## Handle batch job arguments:

nargs = len(sys.argv)
print 'Command line arguments: ' + str(sys.argv)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start = 1000
length_datasets = 200
#######################

# Note: this only sets the random seed for numpy, so if you intend
# on using other modules to generate random numbers, then be sure
# to set the appropriate RNG here

if (nargs==0):
	sim_num = sim_start + 1
	np.random.seed(1330931)
else:
	# Decide on the job number, usually start at 1000:
	sim_num = sim_start + int(sys.argv[1])
	# Set a different random seed for every job number!!!
	np.random.seed(762*sim_num + 1330931)

# Simulation datasets numbered 1001-1200

########################################################################################
########################################################################################



def bayes_logreg(n,y,X,beta_0,Sigma_0_inv,niter=10000,burnin=1000,print_every=1000,retune=100,verbose=False):
	# Stuff...
	return None

#################################################
beta_0 = np.zeros(2)
Sigma_0_inv = np.diag(np.ones(p))
# More stuff...
#################################################

# Read data corresponding to appropriate sim_num:

# Extract X and y:

# Fit the Bayesian model:

# Extract posterior quantiles...

# Write results to a (99 x p) csv file...

# Go celebrate.
 
