#
# Example based on hello_world.py shipped with PyCUDA
#

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

m = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b, int n)
{
  const int i = threadIdx.x;
  if (i<n){
  	dest[i] = a[i] * b[i];
  }
  return;
}
""")

multiply_them = m.get_function("multiply_them")

# Arguments must be numpy datatypes i.e., n = 1000 will not work!

n = np.int32(1000)

# Note: explicit casting to floats

a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)

# Allocate storage for the result:

dest = np.zeros_like(a)

# Launch the kernel:

multiply_them(drv.Out(dest), drv.In(a), drv.In(b), drv.In(n), block=(int(n),1,1), grid=(1,1))

#
# Notes:
#
# -- block and grid specs must be regular python int's, not numpy int's! ;)
# -- drv.Out    specifies what arguments are outputs (i.e., to be copied back)
# -- drv.In     specifies the input-only arguments
# -- drv.InOut  specifies input and output arguments (i.e., overwritten as return value)
#

# Print the differences:

print("sum(abs(dest - a*b)")
print(np.sum(np.abs(dest-a*b)))

print("a[0:9]:")
print(a[0:9])
print("b[0:9]:")
print(b[0:9])
print("dest[0:9]:")
print(dest[0:9])


