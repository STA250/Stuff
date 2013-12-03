## Basic RCUDA Example

To run the example, first compile the kernel using either:

	make random.ptx

or:

	nvcc --ptx -o random.ptx random.cu


Then, run the R script execute the kernel. The basics of the R script are:

+ Load `RCUDA` using `library(RCUDA)`
+ Load the `ptx` module using `m = loadModule("random.ptx")`
+ Extract the kernel using `m$kernel_name`
+ Decide on the block/grid size (as integers, using `L` or `as.integer`)
+ Launch the kernel with `.cuda(kernel_func, args, blockDims, GridDims, outputs)`.


