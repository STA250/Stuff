library(RCUDA)

verbose <- TRUE

if (verbose){
    cat("Setting cuGetContext(TRUE)...\n")
}
cuGetContext(TRUE)

if (verbose){
    cat("Loading module...\n")
}
m <- loadModule("dnorm.ptx")
if (verbose){
    cat("done. Extracting kernel...\n")
}
dnorm_kernel <- m$dnorm_kernel

# Note: kernel looks like this:
# __global__ void dnorm_kernel(float *vals, int N, float mu, float sigma);

if (verbose){
    cat("done. Setting up miscellaneous stuff...\n")
}
N <- 1e8L
x <- rnorm(N)
if (verbose){
    cat("done. Setting mu and sigma...\n")
}
mu <- 0.3
sigma <- 1.5

threads_per_block <- 1024L
bg <- compute_grid(N=N,threads_per_block=threads_per_block)

if (verbose){
    cat("Grid size:\n")
    print(bg$grid_dims)
    cat("Block size:\n")
    print(bg$block_dims)
}

nthreads <- prod(bg$grid_dims)*prod(bg$block_dims) 
if (verbose){
    cat("Total number of threads to launch = ",nthreads,"\n")
}
if (nthreads < N){
    stop("Grid is not large enough...!")
}

cudaDeviceSynchronize()

if (verbose){
    cat("Running CUDA kernel...\n")
}
cu_time <- system.time({
    if (verbose){
        cat("Copying random N(0,1)'s to device...\n")
    }
    cu_copy_to_time <- system.time({mem <- copyToDevice(x)})
    # .cuda(kernel, args, gridDim, blockDim)
    cu_kernel_time <- system.time({.cuda(dnorm_kernel, mem, N, mu, sigma, gridDim=bg$grid_dims, blockDim=bg$block_dims)})
    if (verbose){
         cat("Copying result back from device...\n")
    }
    cu_copy_back_time <- system.time({cu_ret <- copyFromDevice(obj=mem,nels=mem@nels,type="float")})
    # Equivalently:
    #cu_ret <- mem[]
})


cu_kernel2_time <- system.time({y <- .cuda(dnorm_kernel, "x"=x, N, mu, sigma, gridDim=bg$grid_dims, blockDim=bg$block_dims, outputs="x")})

cat("")
r_time <- system.time({
    r_ret <- dnorm(x,mean=mu,sd=sigma)
})
if (verbose){
    cat("done. Finished profile run! :)\n")
}
# Not the best comparison but a rough real-world comparison:
if (verbose){
    cat("CUDA time:\n")
    print(cu_time)
    cat("R time:\n")
    print(r_time)
    cat("Breakdown of CUDA time:\n")
    cat("Copy to device:\n")
    print(cu_copy_to_time)
    cat("Kernel:\n")
    print(cu_kernel_time)
    cat("Copy from device:\n")
    print(cu_copy_back_time)
    cat("Smart kernel version timing:\n")
    print(cu_kernel2_time)
}

# Differences due to floating point vs. double...
tdiff <- sum(abs(cu_ret - r_ret))
if (verbose){
    cat("Diff = ",tdiff,"\n")
    cat("Differences in first few values...\n")
    print(abs(diff(head(cu_ret)-head(r_ret))))
    cat("Differences in last few values...\n")
    print(abs(diff(tail(cu_ret)-tail(r_ret))))
}

# Experiment with clean up:
if (verbose){
    cat("Device memory usage prior to cleanup:\n")
    print(cuMemGetInfo())
    rm(mem)
    gc()
    cat("Device memory usage prior after cleanup:\n")
    print(cuMemGetInfo())
}




