## PyCUDA Example

Illustrates the use of random numbers in `PyCUDA` via multiple methods:

+ Using the CURAND library to directly initialize and generate random numbers inside the kernel (same as in the `RCUDA` examples)
+ Using the `GPUArray` functionality to generate random numbers using `PyCUDA`s extended functionality

Timing comparisons are also made to naive `numpy` approaches.

