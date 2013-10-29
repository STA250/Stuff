# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:20:19 2013

@author: jinzhenfan
"""
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import csv
import pandas as pd
import math


print("Hello")

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
        beta_track.append(beta_curr)
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
        print("v is",v)
        
    print("The covariance after tuning is",v,",","with an acceptance rate of",acpt_rate)
    
#sampling
    beta_curr=beta_0
    beta_track=metropolis_hasting(niter+burnin,beta_init,v,m,y,x,beta_0,Sigma_0_inv)
    beta_samples=beta_track[(burnin+2):(niter+burnin)]
    print("The acceptance rate is",beta_track[0])
#    beta1=beta_samples[1,]
#    beta2=beta_samples[2,]
#    plt.plot[beta_samples[1,]]               
#    effectiveSize(beta1)
#    effectiveSize(beta2)
#    low_margin=np.percentile(beta1,5)
#    high_margin=np.percentile(beta1,95)

#for i in the range(1:200)
#    df = pd.read_csv('blr_data_1'+str(i).zfill(3)'.csv')
df = pd.read_csv('blr_data_1001.csv')
y,m,x= np.matrix(df.y).T,np.matrix(df.n).T,np.matrix([df.X1,df.X2]).T
beta_0=np.matrix([0,0]).T
Sigma_0_inv=np.matrix(np.diag([1,1]))
bayes_logreg(m,y,x,beta_0,Sigma_0_inv,niter=10000,burnin=1000,print_every=1000,retune=100,verbose=True)
              
              