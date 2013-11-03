# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:14:56 2013

@author: jinzhenfan
"""

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
from scipy import stats
import matplotlib.pyplot as plt
import scipy as sp
import csv
import pandas as pd
import math

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
	sim_num = sim_start + int(sys.argv[2])
	# Set a different random seed for every job number!!!
	np.random.seed(762*sim_num + 1330931)

# Simulation datasets numbered 1001-1200

########################################################################################
########################################################################################



def log_density(m,y,x,beta,beta_0,Sigma_0_inv):
    n=len(m)
    m=np.matrix(m)
    y=np.matrix(y)
    x=np.matrix(x)
    beta_0=np.matrix(beta_0)
    beta=np.matrix(beta)
    Sigma_0_inv=np.matrix(Sigma_0_inv)
    s=0
    for i in range(n):
#        print(x[i])
#        print(y[i])
#        print(beta)
#        print(m[i])
        s=s+y[i]*(x[i]*beta)-m[i]*math.log(1+exp(x[i]*beta))
#    print beta.shape
#    print beta_0.T.shape
#    print Sigma_0_inv.shape
#    print s.shape
    log_post=s-(beta-beta_0).T*Sigma_0_inv*[(beta-beta_0)/2][0]

    return log_post

def metropolis_hasting(trial,beta_init,v,m,y,x,beta_0,Sigma_0_inv):
    acpt=0.0   
    acpt_rate=0.0
    beta_curr=beta_init
    beta_track=[0,beta_init]
    for t in range(trial):    
        beta_prosal=np.random.multivariate_normal([0,0], v, 1).T
        log_alpha=log_density(m,y,x,beta_prosal,beta_0,Sigma_0_inv)-log_density(m,y,x,beta_curr,beta_0,Sigma_0_inv)[0][0]
        log_u=math.log(np.random.uniform(low=0.0, high=1.0, size=1))
#        print log_alpha
 #       print log_u
        if log_u < log_alpha[0][0]:
            beta_curr=beta_prosal
            acpt+=1
        else:
            beta_curr=beta_curr
        
        beta_track.append(beta_curr.T)
    acpt_rate=acpt/trial  
#    print("Acceptance is",acpt)
#    print("Acceptance Rate is",str(acpt_rate))
    beta_track[0]=acpt_rate
    return beta_track
    
def confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.stderr(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def bayes_logreg(m,y,x,beta_0,Sigma_0_inv,niter=10000,burnin=1000,print_every=1000,retune=100,verbose=True):
     
    v=np.diag([1,1])
    beta_init=beta_0
    acpt_rate=0
    
# tuning the covariace
    while acpt_rate<0.4 or acpt_rate>0.6:
        acpt_rate=metropolis_hasting(retune,beta_init,v,m,y,x,beta_0,Sigma_0_inv)[0]  
 #       print acpt_rate                  
        if acpt_rate<0.4:
            v=v/exp(1)
        elif acpt_rate>0.6:
            v=v*exp(1) 
#        print("v is",v)
        
    print("The covariance after tuning is",v,",","with an acceptance rate of",acpt_rate)
    
#sampling
    beta_curr=beta_0
    beta_track=metropolis_hasting(niter+burnin,beta_init,v,m,y,x,beta_0,Sigma_0_inv)
    beta_samples=beta_track[(burnin+1):(niter+burnin+1)]
    print("The acceptance rate is",beta_track[0])
    # print(beta_samples)
#    print(beta_samples[0])
#    print(beta_samples[0][0][0])
#    print(beta_samples[0][0][1])
#    print(beta_samples[1])
#    print(beta_samples[2])
#    print(len(beta_samples))
    beta1=np.zeros(niter)
#    print len(beta1)
    beta2=np.zeros(niter)
    for i in range(niter):
        beta1[i]=beta_samples[i][0][0]
        beta2[i]=beta_samples[i][0][1]
#    print beta1
    
#    plt.plot[beta_samples[1,]]               
#    effectiveSize(beta1)
#    effectiveSize(beta2)
    low_margin1=np.percentile(beta1,5)
    high_margin1=np.percentile(beta1,95)
    low_margin2=np.percentile(beta2,5)
    high_margin2=np.percentile(beta2,95)
#    print(low_margin1,high_margin1,low_margin2,high_margin2)
#    print beta1
#    print beta2
    return (beta1,beta2)
         
    

for i in range(200):
    df = pd.read_csv('./data/blr_data_1'+str(i+1).zfill(3)+'.csv')
    #df = pd.read_csv('blr_data_1001.csv')
    y,m,x= np.matrix(df.y).T,np.matrix(df.n).T,np.matrix([df.X1,df.X2]).T
    beta_0=np.matrix([0,0]).T
    Sigma_0_inv=np.matrix(np.diag([1,1]))
    beta1,beta2=bayes_logreg(m,y,x,beta_0,Sigma_0_inv,niter=10000,burnin=1000,print_every=1000,retune=100,verbose=True)
#    print beta1
#    print beta2    
    with open("./results/out_1" + str(i+1).zfill(3)+".csv", 'w') as out:# output file
        writer = csv.writer(out, dialect='excel')
        for row in range(100):        
  #              print np.percentile(beta1,row),np.percentile(beta2,row),"\n" 
                writer.writerow((round(np.percentile(beta1,row),7),round(np.percentile(beta2,row),7)))
    out.closed


# Read data corresponding to appropriate sim_num:

# Extract X and y:

# Fit the Bayesian model:

# Extract posterior quantiles...

# Write results to a (99 x p) csv file...

# Go celebrate.

