/* Header file for CUDA error checking
Include this file in your code and call checkCudaError() after every CUDA API function call or kernel call
checkCudaError() checks if the last CUDA API function call or kernel launch caused an error. 
In case of an error, it prints an error message and exits the program.
*/

#ifndef ERROR_CUH
#define ERROR_CUH

#define checkCudaError() { gpuAssert(__FILE__, __LINE__); }

static inline void gpuAssert(const char *file, int line){
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s \n in file : %s line number : %d", cudaGetErrorString(code), file, line);
        exit(1);
   }
}

#endif
