#include <stdlib.h>
#include <stdio.h>

#define kernel_exec(x,y) x,y; cuda_kernel_error(__FILE__, __LINE__)
inline void cuda_kernel_error(const char* file, int linenum){
	cudaError_t errcode=cudaGetLastError();
	if(errcode!=cudaSuccess){
		printf("Kernel error in file %s line %d: %s\n", file, linenum, cudaGetErrorString(errcode));
		exit(-1);
	}
}

#define kernel_stream_exec(x,y,z,k) x,y,z,k; cuda_kernel_error(__FILE__, __LINE__)
inline void cuda_kernel_stream_error(const char* file, int linenum){
	cudaError_t errcode=cudaGetLastError();
	if(errcode!=cudaSuccess){
		printf("Kernel error in file %s line %d: %s\n", file, linenum, cudaGetErrorString(errcode));
		exit(-1);
	}
}

#define cuda_call(x) cuda_call_check(__FILE__, __LINE__, x)
inline void cuda_call_check(const char* file, int linenum, cudaError_t errcode){
	if(errcode!=cudaSuccess){
		printf("CUDA error in file %s line %d: %s\n", file, linenum, cudaGetErrorString(errcode));
		exit(-1);
	}
}

