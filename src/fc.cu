#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../utils/misc_utils.h"
#include "../utils/thrust_utils.h"
#include "../utils/cublas_utils.h"

using d_type = double;

struct Sigmoid {
    __device__
    float operator()(d_type x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }
};

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    const d_type alpha = 1.0;
    const d_type gemm_beta = 0.0;
    
    const int l1 = 2;
    const int l2 = 2;
    thrust::host_vector<d_type> W1 = {0.15, 0.20, 0.25, 0.30}; // layer 1 weights
    thrust::host_vector<d_type> W2 = {0.40, 0.45, 0.50, 0.55}; // layer 2 weights
    thrust::host_vector<d_type> b1 = {0.35, 0.35}; 
    thrust::host_vector<d_type> b2 = {0.60, 0.60};

    thrust::host_vector<d_type> in = {0.05, 0.10};
    thrust::host_vector<d_type> out(2);

    // print_host_thrust(1, 2, in, 1);
    // print_host_thrust(l1, l2, W1, l1);
    thrust::device_vector<d_type> d_in = in;
    thrust::device_vector<d_type> d_W1 = W1;
    thrust::device_vector<d_type> d_W2 = W2;
    thrust::device_vector<d_type> d_b1 = b1;
    thrust::device_vector<d_type> d_b2 = b2;
    thrust::device_vector<d_type> d_imdte_out = out;
    thrust::device_vector<d_type> d_out(2);

    const int m = 2;
    const int n = 1;
    const int k = 2;

    CUBLAS_CHECK(
        cublasDgemm(
            cublasH, 
            CUBLAS_OP_N, 
            CUBLAS_OP_N, 
            n, k, m, 
            &alpha, 
            thrust::raw_pointer_cast(&d_in[0]), n,
            thrust::raw_pointer_cast(&d_W1[0]), m, 
            &gemm_beta,
            thrust::raw_pointer_cast(&d_imdte_out[0]), n
        )
    );
    // cudaDeviceSynchronize();  // don't need this since we're using cublas stream?

    /*
    two cuBLAS possible calls to add bias vector:
    */

    // using geam will require a beta value of 1 to scale input B properly
    // important: remember that inputs are marked d_* to denote DEVICE matrix
    //            ex. don't forget to use d_b1 (not b1), otherwise CUDA error 700
    //            should probably use h_* for host stuff for best practice
    const int a_rows = 1;
    const int b_cols = 2;
    const d_type geam_beta = 1.0;
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&d_imdte_out[0]), a_rows,
            &geam_beta,
            thrust::raw_pointer_cast(&d_b1[0]), a_rows,
            thrust::raw_pointer_cast(&d_out[0]), a_rows
        )
    );
    // using axpy will require the result to overwrite an input
    // CUBLAS_CHECK(
    //     cublasDaxpy(
    //         cublasH,
    //         2,
    //         &alpha,
    //         thrust::raw_pointer_cast(&d_imdte_out[0]), 1,
    //         thrust::raw_pointer_cast(&d_b1[0]), 1
    //     )
    // );

    thrust::transform(
        d_out.begin(),
        d_out.end(),
        d_out.begin(),
        Sigmoid()
    );
    
    cudaStreamSynchronize(stream);
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    out = d_out;
    print_host_thrust<d_type>(1, 2, out, 1);
    
    return EXIT_SUCCESS;
}