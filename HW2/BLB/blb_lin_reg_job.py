# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:40:05 2013

@author: jinzhen
"""
from __future__ import division
import math
import sys
import numpy as np
import csv


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
        rootfilename = "blb_lin_reg_mini.txt"
else:
        rootfilename = "blb_lin_reg_data.txt"

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
s=5
r=50

job_num=int(sys.argv[nargs-1])
# Find r and s indices:
s_index=job_num//50+1
r_index=job_num%50

if r_index==0:
    r_index=50
    
if r_index==50:
    s_index=s_index-1
# Reset random seed so things differ:
np.random.seed(s_index)

# Bootstrap datasets numbered 1001-1250

########################################################################################
########################################################################################
def read_some_lines(filename,indices,nr,nc,print_every=1000,verbose=True):
        # Storage:
        subset = np.empty(nr*nc)
        subset.shape = (nr,nc)
        # Read file and extract selected rows:
        row_num = 0
        sampled_so_far = 0
        # utility stuff:
        next_ix = indices[0]
        # append value to avoid oob:
        n=0
        indices = np.append(indices,n+1)
        with open(filename, 'rb') as csvfile:
                datareader = csv.reader(csvfile, delimiter=',')
                for current_row in datareader:
                        # Check if needed:
                        if row_num == next_ix:
                                # Yup:
                                if verbose:
                                        print "Sampled row " + str(row_num) + ":"
                                        print current_row
                                # Store values:
                                subset[sampled_so_far,:] = current_row[:]
                                # Increment indices:
                                sampled_so_far += 1
                                next_ix = indices[sampled_so_far]
                        # Increment row counter:
                        row_num += 1
                        # Print progress?
                        if verbose and ((row_num % print_every) == 0):
                                print "Finished row " + str(row_num) + " (sampled " + str(sampled_so_far) + "/" + str(nr) + " rows so far)"
        return subset


# (Weighted) Least Squares Code:

def wls(y, X, w, verbose=False):
        #
        # Fits:
        # y_i = (x_i)^{T} \beta + \epsilon_{i},
        # where \epsilon_{i} \iid N(0,\sigma^{2}/w_{i}).        
        # i.e., each y_{i} represents the mean of w_{i}
        # data points. Setting w=1 corresponds to each
        # data point occuring once.
        #
        # Computes; \hat{\beta} = (X'W^{-1}X)^{-1}X'W^{-1}y
        if verbose:
                print "y (shape = " + str(y.shape) + "):"
                print y
                print "X (shape = " + str(X.shape) + "):"
                print X
                print "w (shape = " + str(w.shape) + "):"
                print w

        Winv = np.diag(1.0/w)
        Winv_y = np.dot(Winv,y)
        Xp = np.transpose(X)
        XpWinv_y = np.dot(Xp,Winv_y)
        XpWinvX = np.dot(Xp,np.dot(Winv,X))
        beta_hat = np.linalg.solve(XpWinvX,XpWinv_y)

        if verbose:
                print "Winv:"
                print Winv
                print "Winv_y:"
                print Winv_y
                print "Xp:"
                print Xp
                print "XpWinv_y:"
                print XpWinv_y
                print "XpWinvX:"
                print XpWinvX
                print "beta_hat:"
                print beta_hat
        return beta_hat



# dataset specs:   
data_cov=1000
data_col=data_cov+1
data_row=1000000
gamma=0.6
sample_row=int(math.floor(data_row**gamma))

#sset parameter of subsets
subset = np.empty(sample_row*data_col)
subset.shape = (sample_row,data_col)
#set beta 
beta_hat = np.empty(data_cov)

# sample indices:

indices=sorted(np.random.choice(data_row,sample_row,replace=False))
subset=read_some_lines(datadir+rootfilename,indices,sample_row,data_col,print_every=1000,verbose=False)

#reset seed
np.random.seed(sim_num)
# draw a bootstrap sample of size n from each subset
entry=np.random.multinomial(data_row,[1/sample_row]*sample_row,size=1)[0]
y=subset[:,-1]
X=subset[:,:-1]
# Fit linear model with weights:
beta_hat=wls(y, X, entry, verbose=False)

# Output file:
outfile = "output/coef_%02d_%02d.txt" % (s_index,r_index)

# Write to file:
with open(outfile,'w') as out:

    output = csv.writer(out, delimiter='\n')
    output.writerow(beta_hat)

out.closed
    

#==============================================================================
# 
# print "========================="
# print "sim_num = " + str(sim_num)
# print "s_index = " + str(s_index)
# print "r_index = " + str(r_index)
# print "========================="
#==============================================================================









