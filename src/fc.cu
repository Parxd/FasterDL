#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../utils/misc_utils.h"
#include "../utils/thrust_utils.h"
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
    
    // 3,4
    // 4,2
    const int l1 = 3;
    const int l2 = 4;
    const int l3 = 2;
    thrust::host_vector<d_type> W1(l1 * l2);
    thrust::host_vector<d_type> W2(l2 * l3);
    thrust::host_vector<d_type> B1(l2);
    thrust::host_vector<d_type> B2(l3);
    thrust::generate(W1.begin(), W1.end(), random_norm<d_type>);
    thrust::generate(W2.begin(), W2.end(), random_norm<d_type>);
    thrust::generate(B1.begin(), B1.end(), random_norm<d_type>);
    thrust::generate(B2.begin(), B2.end(), random_norm<d_type>);
    

    print_host_thrust(l1, l2, W1, l1);

    return EXIT_SUCCESS;
}