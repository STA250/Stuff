import sys
import csv
import numpy as np
import wls
import read_some_lines as rsl

print "=============================="
print "Python version:"
print str(sys.version)
print "Numpy version:"
print str(np.version.version)
print "=============================="

mini = False
verbose = False

datadir = "/home/pdbaines/data/"
outpath = "output/"

# mini or full?
if mini:
	rootfilename = "blb_lin_reg_mini"
else:
	rootfilename = "blb_lin_reg_data"

########################################################################################
## Handle batch job arguments:

nargs = len(sys.argv)
print 'Command line arguments: ' + str(sys.argv)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start = 1000
length_datasets = 250
#######################

# Note: this only sets the random seed for numpy, so if you intend
# on using other modules to generate random numbers, then be sure
# to set the appropriate RNG here

if (nargs==1):
	sim_num = sim_start + 1
	sim_seed = 1330931
else:
	# Decide on the job number, usually start at 1000:
	sim_num = sim_start + int(sys.argv[nargs-1])
	# Set a different random seed for every job number!!!
	sim_seed = 762*sim_num + 1330931

# Set the seed:
np.random.seed(sim_seed)

# Bootstrap datasets numbered 1001-1250

########################################################################################
########################################################################################

# Find r and s indices:
r_index = ...
s_index = ...

print "========================="
print "sim_num = " + str(sim_num)
print "s_index = " + str(s_index)
print "r_index = " + str(r_index)
print "========================="

# filename:
# dataset specs:
# Set random seed so indices are same for each s:
# sample indices:
# Reset random seed so things differ:
# Take the subset:
# Bootstrap to full datasize:
# Fit linear model with weights:
# Output file:
# Write to file:


