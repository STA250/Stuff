NVCC=/usr/local/cuda/bin/nvcc
BIN_OPTS= -m64 -gencode arch=compute_10,code=sm_10
PTX_OPTS= -m64 -gencode arch=compute_30,code=sm_30
# -m64 -arch sm_21

%.ptx: %.cu
	$(NVCC) --ptx $(PTX_OPTS) -o $@ $<

%.cubin: %.cu GNUmakefile
	$(NVCC) -cubin $(BIN_OPTS)  -o $@ $<

%.fatbin: %.cu GNUmakefile
	$(NVCC) -fatbin $(BIN_OPTS) -o $@ $<

