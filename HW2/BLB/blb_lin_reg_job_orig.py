# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:40:05 2013

@author: jinzhen
"""
from __future__ import division
import numpy as np
import csv
import math
#import sys





##
# filename: name of csv file
# indices: array of row numbers to select
# nr: number of rows to sample i.e., len(indices)
# nc: number of cols in file
# print_every: number of lines per status update
# verbose: True/False, level of verbosity
##

def read_some_lines_csv(filename,indices,nr,nc,print_every=1000,verbose=True):
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



#parameters     
data_cov=40
data_col=data_cov+1
#n
data_row=10000
gamma=0.6
#b
sample_row=int(math.floor(data_row**gamma))
s=5
r=50
#sample bag of bootstrap subsets
subset = np.empty(data_col*sample_row*s)
subset.shape = (s,sample_row,data_col)
sub_SE=np.empty(s*data_cov)
sub_SE.shape=(s,data_cov)

for j in range(s):

#    bootstrap = np.empty(data_col*data_row*s)
#    bootstrap.shape = (s,data_row,data_col)
    beta_hat = np.empty(r*data_cov)
    beta_hat.shape= (r,data_cov)
    indices=sorted(np.random.choice(data_row,sample_row,replace=False))
    subset[j]=read_some_lines_csv('blb_lin_reg_mini.txt',indices,sample_row,data_col,print_every=1000,verbose=False)
    
    for k in range(r):
        # draw a bootstrap sample of size n from each subset
        entry=np.random.multinomial(data_row,[1/sample_row]*sample_row,size=1)[0]
#        bootstrap[j]=np.repeat(subset[j],entry)
        y=subset[j][:,-1]
        X=subset[j][:,:-1]
        beta_hat[k]=wls(y, X, entry, verbose=False)
        
    #calculate standard deviation of each bootstrap subsets
    
    sub_SE[j]=np.std(beta_hat,axis=0)
    
# calculate the standard deviation of all 
SE=np.mean(sub_SE,axis=0)
print SE
        





