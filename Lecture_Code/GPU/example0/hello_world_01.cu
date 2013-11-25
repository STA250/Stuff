#include <stdio.h>

// Note: Needs compute capability > 2.0, so compile with:
// nvcc hello_world_01.cu -arch=compute_20 -code=sm_20,compute_20 -o hello_world_01.out
// Other notes: can have trouble when N is large...
// Default buffer is ~8MB
// See hello_world_02.cu for details.

#include <cuda.h>
#include <cuda_runtime.h>

#define N 20000
#define GRID_D1 20
#define GRID_D2 2
#define BLOCK_D1 512
#define BLOCK_D2 1
#define BLOCK_D3 1

__global__ void hello(void)
{
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;
    if (idx < N){  
        printf("Hello world! My block index is (%d,%d) [Grid dims=(%d,%d)], 3D-thread index within block=(%d,%d,%d) => thread index=%d\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, threadIdx.x, threadIdx.y, threadIdx.y, idx);
    } else {
        printf("Hello world! My block index is (%d,%d) [Grid dims=(%d,%d)], 3D-thread index within block=(%d,%d,%d) => thread index=%d [### this thread would not be used for N=%d ###]\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, threadIdx.x, threadIdx.y, threadIdx.y, idx, N);
    }
}


int main(int argc,char **argv)
{
    const dim3 blockSize(BLOCK_D1, BLOCK_D2, BLOCK_D3);
    const dim3 gridSize(GRID_D1, GRID_D2, 1);
    int nthreads = BLOCK_D1*BLOCK_D2*BLOCK_D3*GRID_D1*GRID_D2;
    if (nthreads < N){
        printf("\n============ NOT ENOUGH THREADS TO COVER N=%d ===============\n\n",N);
    } else {
        printf("Launching %d threads (N=%d)\n",nthreads,N);
    }
    
    // launch the kernel
    hello<<<gridSize, blockSize>>>();
    
    // Need to flush prints...
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr){
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    } else {
        printf("kernel launch success!\n");
    }
    
    printf("That's all!\n");

    return 0;
}




