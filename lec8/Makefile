todo: matrixAdd matrixMul
matrixAdd: matrixAdd.cu
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 matrixAdd.cu -o matrixAdd
matrixMul: matrixMul.cu
	nvcc -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 matrixMul.cu -o matrixMul
clean:
	rm matrixAdd matrixMul
