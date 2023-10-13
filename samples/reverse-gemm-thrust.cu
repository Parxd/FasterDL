#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../utils/cublas_utils.h"

using d_type = double;

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    const d_type alpha = 1.0;
    const d_type beta = 0.0;

    const int m = 3;
    const int n = 2;
    const int k = 2;
    thrust::host_vector<d_type> W = {1, 2, 3, 4, 5, 6};
    thrust::host_vector<d_type> X = {10, 11, 12, 13, 14, 15};
    thrust::host_vector<d_type> Z(n * k);
    thrust::device_vector<d_type> d_W = W;
    thrust::device_vector<d_type> d_X = X;
    thrust::device_vector<d_type> d_Z = Z;
    // Reverse GEMM to compute (row-major) transposed Z
    CUBLAS_CHECK(
        cublasDgemm(
            cublasH, 
            CUBLAS_OP_N, 
            CUBLAS_OP_N, 
            n, k, m, 
            &alpha, 
            thrust::raw_pointer_cast(&d_X[0]), n,
            thrust::raw_pointer_cast(&d_W[0]), m, 
            &beta,
            thrust::raw_pointer_cast(&d_Z[0]), n
        )
    );
    cudaDeviceSynchronize();
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    print_device_thrust<d_type>(n, k, d_Z, n);
    return EXIT_SUCCESS;
}
