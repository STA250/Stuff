library(RCUDA)

cat("Setting cuGetContext(TRUE)...\n")
cuGetContext(TRUE)
cat("done. Profiling CUDA code...\n")

cat("Loading module...\n")
m = loadModule("random.ptx")
cat("done. Extracting kernels...\n")
k_setup = m$setup_kernel
k_rnorm = m$runif_kernel
k_runif = m$rnorm_kernel
k_allinone = m$rnorm_all_in_one_kernel

cat("done. Setting up miscellaneous stuff...\n")
N <- 250000L # 1e6L fails on my mac... :/
verbose <- FALSE

# Uniform parameters:
lo <- -1.0
hi <- 1.0

# Normal parameters:
mu <-  -0.3
sigma <-  1.5

# Poison parameters:
lambda <- 1.0

# if...
# N = 1,000,000
# => 1954 blocks of 512 threads will suffice
# => (62 x 32) grid, (512 x 1 x 1) blocks

# Fix block dims:
threads_per_block <- 512L
block_dims <- c(threads_per_block, 1L, 1L)
grid_d1 <- floor(sqrt(N/threads_per_block))
grid_d2 <- ceiling(N/(grid_d1*threads_per_block))
grid_dims <- c(grid_d1, grid_d2, 1L)

cat("Grid size:\n")
print(grid_dims)
cat("Block size:\n")
print(block_dims)

nthreads <- prod(grid_dims)*prod(block_dims) 
cat("Total number of threads to launch = ",nthreads,"\n")
if (nthreads < N){
    stop("Grid is not large enough...!")
}

# Need to setup RNG (this doesn't work yet since
# getElementSize() isn't defined for curandState...
# In the meantime, on my Mac: 
# sizeof(curandState) = 48
cat("Allocating memory on device for curandStates...\n")
cu_rng_alloc_time <- system.time({
    rng_states <- cudaMalloc(elType="curandState", numEls=N, sizeof=48L) 
})

#######################################################################################################
# All-in=one kernels

# Automated copying version (no manual copying to/from device):
cat("Launching all-in-one CUDA kernel (v1)...\n")
cu_automated_time <- system.time({
    x_double <- rep(0.0,N)
    cu_rnorm_v1 <- .cuda(k_allinone, "x"=x_double, N, mu, sigma, gridDim=grid_dims, blockDim=block_dims, outputs="x")
})

# Manual copying version:
cat("Launching all-in-one CUDA kernel (v2)...\n")
cu_manual_total_time <- system.time({
cu_all_in_one_time <- system.time({
    x_d_mem <- copyToDevice(x_double) 
    .cuda(k_allinone, x_d_mem, N, mu, sigma, gridDim=grid_dims, blockDim=block_dims)
})
cat("Copying result back from device...\n")
cu_all_in_one_copy_time <- system.time({
    cu_rnorm_v2 <- copyFromDevice(obj=x_d_mem,nels=x_d_mem$nels,type="float")
})
})
cat("First few values (v1)...\n")
print(head(cu_rnorm_v1))
cat("First few values (v1)...\n")
print(head(cu_rnorm_v2))
cat(sprintf("(mu = %g, sigma=%g, n=%d)\n",mu,sigma,N))
cat(sprintf(".cuda V1: (Mean = %g, SD=%g)\n",mean(cu_rnorm_v1),sd(cu_rnorm_v1)))
cat(sprintf(".cuda V2: (Mean = %g, SD=%g)\n",mean(cu_rnorm_v1),sd(cu_rnorm_v2)))

#######################################################################################################

# Initializing RNG's...
cat("Launching CUDA kernel for RNG setup...\n")
cu_init_time <- system.time({
    .cuda(k_setup, rng_states, N, as.integer(verbose), gridDim=grid_dims, blockDim=block_dims)
    cudaDeviceSynchronize()
})

# Call RNGs...
cat("Launching runif CUDA kernel...\n")
cu_runif_time <- system.time({
    .cuda(k_runif, rng_states, x_d_mem, N, lo, hi, gridDim=grid_dims, blockDim=block_dims)
})

cat("Copying result back from device...\n")
cu_runif_copy_time <- system.time({
    cu_runif_x = copyFromDevice(obj=x_d_mem,nels=x_d_mem@nels,type="float")
})

r_runif_time <- system.time({
    r_runif_x <- runif(n=N,min=lo,max=hi)
})

# Call RNGs...
cat("Launching rnorm CUDA kernel...\n")
cu_rnorm_time <- system.time({
    .cuda(k_rnorm, rng_states, x_d_mem, N, mu, sigma, gridDim=grid_dims, blockDim=block_dims)
})

cat("Copying result back from device...\n")
cu_rnorm_copy_time <- system.time({
    cu_rnorm_x = copyFromDevice(obj=x_d_mem,nels=x_d_mem@nels,type="float")
})

r_rnorm_time <- system.time({
    r_rnorm_x <- rnorm(n=N,mean=mu, sd=sigma)
})

tlist <- paste(c("cu_rng_alloc","cu_init",
                 "cu_automated","cu_manual_total",
                 "cu_all_in_one","cu_all_in_one_copy",
                 "cu_runif","cu_runif_copy","r_runif",
                 "cu_rnorm","cu_rnorm_copy","r_rnorm"), "time", sep = "_")

cat("===========================\n")
cat("Timing information:\n")
invisible(sapply(tlist, function(obj) {
     cat(paste(obj, ":\n", sep = ""))
     print(get(obj))
   }))

cat("===========================\n")



