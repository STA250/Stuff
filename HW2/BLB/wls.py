# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:02:01 2013

@author: paul
"""

import numpy as np

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


