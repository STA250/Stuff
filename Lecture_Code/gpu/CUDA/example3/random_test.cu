#include <stdio.h>

//
// STANDALONE curand example code
//
// Note: Needs compute capability > 2.0, so compile with:
// nvcc random_test.cu -arch=compute_20 -code=sm_20,compute_20 -o random_test.out
// Other notes: can have trouble when N is large...
// Default buffer is ~8MB
// See hello_world_02.cu for details.

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define N 200
#define GRID_D1 4
#define BLOCK_D1 64

__global__ void rnorm_all_in_one_kernel(float *vals, int n, float mu, float sigma)
{
    // 1D-1D thread indexing for simplicity...
    int myblock = blockIdx.x;
    int blocksize = blockDim.x;
    int subthread = threadIdx.x;
    int idx = myblock * blocksize + subthread;

    // Setup the RNG:
    curandState rng_state;
    curand_init(9131 + idx*17, 0, 0, &rng_state);

    printf("idx=%d, mu=%f, sigma=%f\n",idx,mu,sigma);

	if (idx < n) {
	    vals[idx] = mu + sigma * curand_normal(&rng_state);
	}
    return;
}

int main(int argc,char **argv)
{
    printf("Getting some info about CUDA devices on your system...\n");
    int devCount;
    cudaGetDeviceCount(&devCount);
    if (devCount == 1){
        printf("There is %d CUDA device on your system.\n", devCount);
    } else {
        printf("There are %d CUDA devices on your system.\n", devCount);
     }

    // Iterate through devices
    for (int i=0; i<devCount; i++)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
	printf("Name:  %s\n", devProp.name);
    }
    printf("\n");

    int rtv;
    cudaRuntimeGetVersion(&rtv);
    int dv;
    cudaDriverGetVersion(&dv);

    printf("============================================\n");
    printf("CUDA Driver Version:  %d\n",dv);
    printf("CUDA Runtime Version: %d\n",rtv);
    printf("============================================\n\n");

    const dim3 blockSize(BLOCK_D1, 1, 1);
    const dim3 gridSize(GRID_D1, 1, 1);
    int nthreads = BLOCK_D1*GRID_D1;
    if (nthreads < N){
        printf("\n============ NOT ENOUGH THREADS TO COVER N=%d ===============\n\n",N);
    } else {
        printf("Launching %d threads (N=%d)\n",nthreads,N);
    }
    
    int n=N;
    float mu=0.0;
    float sigma=1.0;
    float *gpu_x;
    float *x = (float *) malloc(n*sizeof(float));
    cudaError_t cudaStat = cudaMalloc((void **)&gpu_x, n*sizeof(float));
    if (cudaStat){
        printf(" value = %d : Memory Allocation on GPU Device failed\n", cudaStat);
    } else {
        printf("done. Launching kernel...\n");
    }
    cudaError_t cudaerr;
    cudaerr = cudaDeviceSynchronize();
  
    // launch the kernel
    rnorm_all_in_one_kernel<<<gridSize, blockSize>>>(gpu_x,n,mu,sigma);
    cudaerr = cudaDeviceSynchronize();
  
    cudaStat = cudaMemcpy(x, gpu_x, n*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStat){
        printf(" value = %d : Memory copy from GPU Device failed\n", cudaStat);
    } else {
        printf("done. Tidying up...\n");
    }
    
    printf("Result:\n");
    for (int i=0; i<n; i++){
        printf("%f\t",x[i]);
        if (i%8 == 0){
            printf("\n");
        }
    }
    printf("\n\n");

    // Need to flush prints...
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr){
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    } else {
        printf("kernel launch success!\n");
    }
    
    printf("That's all!\n");

    return 0;
}




