#
# Example based on hello_world.py shipped with PyCUDA
#

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
import numpy as np
from pycuda.compiler import SourceModule

##
# From: http://documen.tician.de/pycuda/array.html 
#
# Warning:
# The following classes are using random number generators that run on the GPU.
# Each thread uses its own generator. Creation of those generators requires 
# more resources than subsequent generation of random numbers. After 
# experiments it looks like maximum number of active generators on Tesla 
# devices (with compute capabilities 1.x) is 256. Fermi devices allow for 
# creating 1024 generators without any problems. If there are troubles with 
# creating objects of class PseudoRandomNumberGenerator or 
# QuasiRandomNumberGenerator decrease number of created generators (and 
# therefore number of active threads).
##

m = SourceModule("""
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

extern "C" {
__global__ void rnorm_kernel(float *dest, float mu, float sigma, int n)
{
    int myblock = blockIdx.x;   // 1D-grid
    int blocksize = blockDim.x; // 1D-block
    int subthread = threadIdx.x;
    int idx = myblock * blocksize + subthread;
    // Setup the RNG:
    curandState rng_state;
    curand_init(93131 + 76*idx, 0, 0, &rng_state);
    if (idx < n) {
        dest[idx] = mu + sigma * curand_normal(&rng_state);
    }
    return;
}
}
""",include_dirs=['/usr/local/cuda/include/'],no_extern_c=1)

rnorm = m.get_function("rnorm_kernel")

# Arguments must be numpy datatypes i.e., n = 1000 will not work!

n = np.int32(1e8)

# Threads per block and number of blocks:
tpb = int(512)
nb = int(1 + (n/tpb))

# mu and sigma vectorized:
mu = np.float32(1.0)
sigma = np.float32(0.1) 

# Allocate storage for the result:
dest = np.zeros(n).astype(np.float32)

# Create two timers:
start = drv.Event()
end = drv.Event()

# Launch the kernel:

start.record()
rnorm(drv.Out(dest),mu,sigma,n,block=(tpb,1,1), grid=(nb,1))
end.record() # end timing
# calculate the run length
end.synchronize()
gpu_secs = start.time_till(end)*1e-3
print("SourceModule time: %f" % gpu_secs)

rng = curandom.XORWOWRandomNumberGenerator()  # be kind and exclude initialization
start.record()
gpu_res = rng.gen_normal(n,dtype=np.float32)  # lives on the device
dest2 = np.add(np.multiply(gpu_res.get(),sigma),mu)  # copy and scale
end.record() # end timing
# calculate the run length
end.synchronize()
gpu2_secs = start.time_till(end)*1e-3
print("    GPUArray time: %f" % gpu2_secs)

start.record()
# Numpy version:
start.record()
host = np.random.normal(size=n,loc=mu,scale=sigma)
end.record() # end timing
# calculate the run length
end.synchronize()
cpu_secs = start.time_till(end)*1e-3
print("       Numpy time: %fs" % cpu_secs)
print("\n")

# Mean and SD (np.float32 cannot handle sum of 1e8 elts):
print("mean (GPU):    %f" % np.mean(dest.astype(np.float64)))
print("  SD (GPU):    %f" % np.std(dest.astype(np.float64)))
print("mean (GPU v2): %f" % np.mean(dest2.astype(np.float64)))
print("  SD (GPU v2): %f" % np.std(dest2.astype(np.float64)))
print("mean (CPU):    %f" % np.mean(host))
print("  SD (CPU):    %f" % np.std(host))


