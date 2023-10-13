#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../utils/thrust_utils.h"

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
    d_type W[k * m]{1, 2, 3, 4, 5, 6};
    d_type X[m * n]{10, 11, 12, 13, 14, 15};
    d_type Z[k * n];

    d_type* d_W;
    d_type* d_X;
    d_type* d_Z;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_W), sizeof(W)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_X), sizeof(X)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Z), sizeof(Z)));
    CUDA_CHECK(cudaMemcpyAsync(d_W, W, sizeof(W), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_X, X, sizeof(X), cudaMemcpyHostToDevice, stream));
    
    // Reverse GEMM to compute (row-major) transposed Z
    CUBLAS_CHECK(
        cublasDgemm(
            cublasH, 
            CUBLAS_OP_N, 
            CUBLAS_OP_N, 
            n, k, m, 
            &alpha, 
            d_X, n,
            d_W, m, 
            &beta,
            d_Z, n
        )
    );

    CUDA_CHECK(cudaMemcpyAsync(Z, d_Z, sizeof(Z), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Z));
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    print_matrix(k, n, Z, k);

    return EXIT_SUCCESS;
}