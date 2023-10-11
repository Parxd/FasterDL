#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../include/cublas_utils.h"

using d_type = double;

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    const d_type alpha = 1.0;
    const d_type beta = 0.0;

    int m = 1;
    int n = 1;
    int k = 3;
    d_type W[m * k]{0.5, 0.5, 0.5};
    d_type X[k * n]{2,   2,   2  };
    d_type Z[m * n]{0};

    d_type* d_W;
    d_type* d_X;
    d_type* d_Z;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(W)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(X)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Z), sizeof(Z)));
    CUDA_CHECK(cudaMemcpyAsync(d_W, W, sizeof(W), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_X, X, sizeof(X), cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(
        cublasDgemm(
            cublasH, 
            CUBLAS_OP_N, 
            CUBLAS_OP_N, 
            m, n, k, 
            &alpha, 
            d_W, 1, 
            d_X, 3, 
            &beta, 
            d_Z, 1
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

    print_matrix(m, n, Z, 1);

    return EXIT_SUCCESS;
}