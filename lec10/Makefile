todo: reduction_atomic_1 reduction_atomic_2 reduction_atomic_3 reduction_shared thrust_sort thrust_sort_by_key thrust_reduce
reduction_atomic_1: reduction_atomic_1.cu
	nvcc -O3 -gencode arch=compute_50,code=sm_50 reduction_atomic_1.cu -o reduction_atomic_1
reduction_atomic_2: reduction_atomic_2.cu
	nvcc -O3 -gencode arch=compute_50,code=sm_50 reduction_atomic_2.cu -o reduction_atomic_2
reduction_atomic_3: reduction_atomic_3.cu
	nvcc -O3 -gencode arch=compute_50,code=sm_50 reduction_atomic_3.cu -o reduction_atomic_3
reduction_shared: reduction_shared.cu
	nvcc -O3 -gencode arch=compute_50,code=sm_50 reduction_shared.cu -o reduction_shared
thrust_sort: thrust_sort.cu
	nvcc -O3 -gencode arch=compute_50,code=sm_50 thrust_sort.cu -o thrust_sort
thrust_sort_by_key: thrust_sort_by_key.cu
	nvcc -O3 -gencode arch=compute_50,code=sm_50 thrust_sort_by_key.cu -o thrust_sort_by_key
thrust_reduce: thrust_reduce.cu
	nvcc -O3 -gencode arch=compute_50,code=sm_50 thrust_reduce.cu -o thrust_reduce
clean:
	rm reduction_atomic_1 reduction_atomic_2 reduction_atomic_3 reduction_shared thrust_sort thrust_sort_by_key thrust_reduce
