#
# Example based on dnorm from RCUDA
# Timing code from http://wiki.tiker.net/PyCuda/Examples/SimpleSpeedTest
#

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import scipy as sp
from scipy.stats import norm
from pycuda.compiler import SourceModule

# Versions:
drv.get_version()
drv.get_driver_version()

m = SourceModule("""
#include <stdio.h>
__global__ void dnorm_kernel(float *vals, float *x, int N, float mu, float sigma, int dbg)
{
    int myblock = blockIdx.x;   // 1D-grid
    int blocksize = blockDim.x; // 1D-block
    int subthread = threadIdx.x;
    int idx = myblock * blocksize + subthread;
    if (idx < N) {
        if (dbg){
            printf("thread idx: %04d\\t x[%d] = %f\\t (n=%d,mu=%f,sigma=%f)\\n",idx,idx,x[idx],N,mu,sigma);
        }
        float std = (x[idx] - mu)/sigma;
        float e = exp( - 0.5 * std * std);
        vals[idx] = e / ( sigma * sqrt(2 * 3.141592653589793));
    } else {
        if (dbg){
            printf("thread idx: %04d\\t (>=N=%d)\\n",idx,N);
        }
    }
    return;
}
""")

dnorm = m.get_function("dnorm_kernel")

# Arguments must be numpy datatypes i.e., n = 1000 will not work!

n = np.int32(1e8)

# Threads per block and number of blocks:
tpb = int(512)
nb = int(1 + (n/tpb))

# Note: explicit casting to floats
print("Generating random normals...")
a = np.random.normal(size=n).astype(np.float32)

# Evaluate at N(-0.2,0.8)

mu = np.float32(-0.2)
sigma = np.float32(0.8)
dbg = False # True
verbose = np.int32(dbg)

# Allocate storage for the result:

dest = np.zeros_like(a)

# Create two timers:
start = drv.Event()
end = drv.Event()

# Launch the kernel (dest must be out, a in):
print("Running GPU code...")
start.record()
dnorm(drv.Out(dest), drv.In(a), n, mu, sigma, verbose, block=(tpb,1,1), grid=(nb,1))
end.record() # end timing
# calculate the run length
end.synchronize()
gpu_secs = start.time_till(end)*1e-3
print("SourceModule time:")
print("%fs" % gpu_secs)

# Scipy version:
print("Running Scipy CPU code...")
start.record()
host = norm.pdf(a,loc=mu,scale=sigma)
end.record() # end timing
# calculate the run length
end.synchronize()
sp_secs = start.time_till(end)*1e-3
print "scipy time:"
print "%fs" % sp_secs

# Print the differences:

print("Differences:")
print(np.sum(np.abs(dest-host)))



