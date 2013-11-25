/* Cuda Program for finding cos(0), cos(1*2*pi/N), ... , cos((N-1)*2*pi/N) */

// nvcc CUDA_example_01.cu -use_fast_math -o CUDA_example_01.out

/* --------------------------- header secton ----------------------------*/

#include<stdio.h>
#include<cuda.h>

#define PRINT_RESULT 1
#define COS_THREAD_CNT 512
#define N 1000000
#define TWO_PI 6.283185

/* --------------------------- target code ------------------------------*/

struct cosParams {
	float *arg;
	float *res;
	int n;
};

// CUDA kernel:

__global__ void cos_main(struct cosParams params)
{
	// Computes:
	// res[threadIdx.x + 0*COS_THREAD_CNT] = cos(threadIdx.x + 0*COS_THREAD_CNT)
	// res[threadIdx.x + 1*COS_THREAD_CNT] = cos(threadIdx.x + 1*COS_THREAD_CNT)
	// res[threadIdx.x + 2*COS_THREAD_CNT] = cos(threadIdx.x + 2*COS_THREAD_CNT)
	// ...etc...
	//
	// if COS_THREAD_CNT = 1, this computes all of the cosines in one go,
	// otherwise, it spreads it out across threads...
	//
	// Choice of either:
	//
	// __cosf:
	// Calculate the fast approximate cosine of the input argument x, measured in radians.
	//
	// cosf:
	// Calculate the cosine of the input argument x (measured in radians).

	int i;
	for (i=threadIdx.x; i<params.n; i+=COS_THREAD_CNT){
		// params.res[i] = __cosf(params.arg[i]);
		params.res[i] = cosf(params.arg[i]);
	}
	return;
}

/* --------------------------- host code ------------------------------*/
int main (int argc, char *argv[])
{
	printf("\nBeginning CUDA example code on the CPU...\n");

	printf("Getting some info about CUDA devices on your system...\n");

	int i, devCount;
	cudaGetDeviceCount(&devCount);
	
	if (devCount == 1){
    		printf("There is %d CUDA device on your system.\n", devCount);
 	} else {
 		printf("There are %d CUDA devices on your system.\n", devCount);
 	}

    	// Iterate through devices
    	for (i=0; i<devCount; i++)
    	{
        	// Get device properties
        	printf("\nCUDA Device #%d\n", i);
        	cudaDeviceProp devProp;
        	cudaGetDeviceProperties(&devProp, i);
		printf("Name:  %s\n", devProp.name);

	}
	printf("\n");

	// Begin cosine code:
	cudaError_t cudaStat;
	float* gpu_res = 0;
	float* gpu_arg = 0;

	printf("Allocating memory on the CPU... (n=%d)\n",(int)N);

	// Allocate the vector 1,...,N:
	float *arg = (float *) malloc(N*sizeof(float));
	// Allocate vector of length N to store the result:
	float *res = (float *) malloc(N*sizeof(float));
	int print_every = (int)((double)N/10.0);

	struct cosParams funcParams;

	// Populate arg with 1,...,N
	for(i=0; i<N; i++){
		arg[i] = (float)i*(float)TWO_PI/(float)N;
	}

	printf("Allocating memory on the GPU... (n=%d)\n",(int)N);

	// Key function:
	// cudaError_t cudaMalloc(void ** devPtr, size_t size);
	
	// Allocate N floats on the GPU for the argument 0,1,...,N-1, and make gpu_arg a pointer to that memory:
	// NOTE: sizeof(gpu_arg[0]) is equivalent to sizeof(float) here
	cudaStat = cudaMalloc((void **)&gpu_arg, N*sizeof(gpu_arg[0]));
	if (cudaStat){
		printf(" value = %d : Memory Allocation on GPU Device failed\n", cudaStat);
	} else {
		printf("done. Allocating more memory on the GPU...\n");
	}

	// Allocate N floats on the GPU to store the result, and make gpu_res a pointer to that memory:
	// NOTE: sizeof(gpu_res[0]) is equivalent to sizeof(float) here
	cudaStat = cudaMalloc((void **)&gpu_res, N*sizeof(gpu_res[0]));
	if (cudaStat){
		printf(" value = %d : Memory Allocation on GPU Device failed\n", (int)cudaStat);
	} else {
		printf("done again. Copying stuff from host (CPU) to device (GPU)...\n");
	}

	// Key function:
	// cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind);	

	// Copy the vector 0,1,...,N-1 from arg (on the host) to gpu_arg (on the device)
	// NOTE: sizeof(arg[0]) is equivalent to sizeof(float) here
    cudaStat = cudaMemcpy(gpu_arg, arg, N*sizeof(arg[0]), cudaMemcpyHostToDevice);
	if (cudaStat){
		printf(" Memory Copy from Host to Device failed.\n");
	} else {
		printf("successful.\n");
	}

	// Set up the parameters for the GPU kernel:
	funcParams.res = gpu_res;
	funcParams.arg = gpu_arg;
	funcParams.n = N;

	printf("Launching kernel on GPU...\n");

	// Launch the GPU kernel...

	// Key code:
	// KernelFunction<<<dimGrid, dimBlock>>>(args);

	cos_main<<<1,COS_THREAD_CNT>>>(funcParams);

	printf("GPU computations finished. Copying result back to CPU...\n");

	// Copy the vector cos(0),cos(1),...,cos(N-1) from gpu_res (on the device) to res (on the host)
	cudaStat = cudaMemcpy(res, gpu_res, N*sizeof(gpu_res[0]), cudaMemcpyDeviceToHost);

	if (cudaStat){
		printf(" Memory Copy from Device to Host failed.\n");
	} else {
		printf("Copy back successful. Printing result...\n");
	}

	if (PRINT_RESULT){

		// Print the result?
		for(i=0; i < N; i++ ){
			if (i%print_every == 0 ){
				printf("cos(%f) = %f\n", arg[i], res[i] );
			}
		}

	}

    printf("Freeing memory...\n");
    free(arg);
    free(res);
    cudaFree(gpu_arg);
    cudaFree(gpu_res);

	printf("\n\nFinished. :)\n\n");

	return 0;
}


