#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "../include/cublas_utils.h"

using d_type = double;

void forward() {

}

void backward() {
	
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    const d_type alpha = 1.0;
    const d_type beta = 0.0;

    // Layer 1 - 3 nodes
    // Layer 2 - 4 nodes
    // Layer 3 - 2 nodes
    
    
    return EXIT_SUCCESS;
}