
# Run from:
# ipython --pylab
# to use plot command as listed, else:

import scipy
from scipy import stats
import numpy as np
import matplotlib.pyplot as mp

# numpy example:

np.random.binomial(10,0.5)
np.random.binomial([2,10,100],[0.1,0.5,0.9])

# scipy example, R-like with (r,d,p,q) ==> (rvs,pmf,cdf,ppf)

p = np.array([0.1,0.5,0.9])
n = np.array([2,10,100])
print n.dtype
print p.dtype

y = scipy.stats.binom.rvs(n,p)
print str(y)

p_x = scipy.stats.binom.pmf(y,n,p)
print p_x

F_x = scipy.stats.binom.cdf(y,n,p)
print F_x

q_x = scipy.stats.binom.ppf([0.5,0.5,0.5],n,p)
print q_x

# Other numpy examples:
np.random.normal()
np.random.uniform()

# Other scipy examples:
scipy.stats.poisson.rvs(1,size=100)
scipy.stats.gamma.rvs(1,1,size=100)
scipy.stats.uniform(size=100)
y = scipy.stats.norm.rvs(size=100)
# etc.

# Simple histogram:
mp.hist(y)
mp.show()


