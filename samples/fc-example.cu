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
    __device__ d_type operator()(d_type x) const {
        return 1 / (1 + std::exp(-x));
    }
};
struct MSE {
    __device__ d_type operator()(d_type x) const {
        return x * x * 0.5;
    }
};

/*
for the sake of truly understanding how partial derivatives are calculated
and to see what computations can be reused when building up the computation
graph engine later, this is an example of a function-less, object-less forward &
backward run of a fully-connected neural net (2, 2)

uses example values from: 
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
*/
int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    const d_type alpha = 1.0;
    const d_type gemm_beta = 0.0;
    const d_type geam_beta = 1.0;
    const d_type negate_geam_beta = -1.0;

    thrust::host_vector<d_type> W1 = {0.15, 0.20, 0.25, 0.30}; // layer 1 weights
    thrust::host_vector<d_type> W2 = {0.40, 0.45, 0.50, 0.55}; // layer 2 weights
    thrust::host_vector<d_type> b1 = {0.35, 0.35}; 
    thrust::host_vector<d_type> b2 = {0.60, 0.60};

    thrust::host_vector<d_type> in = {0.05, 0.10};
    thrust::host_vector<d_type> out1(2);
    thrust::host_vector<d_type> out2(2);

    // print_host_thrust(1, 2, in, 1);
    // print_host_thrust(l1, l2, W1, l1);
    thrust::device_vector<d_type> d_in = in;
    thrust::device_vector<d_type> d_W1 = W1;
    thrust::device_vector<d_type> d_W2 = W2;
    thrust::device_vector<d_type> d_b1 = b1;
    thrust::device_vector<d_type> d_b2 = b2;

    thrust::device_vector<d_type> d_imdte_out1 = out1;
    thrust::device_vector<d_type> d_imdte_out2 = out2;
    thrust::device_vector<d_type> d_out1(2);
    thrust::device_vector<d_type> d_out2(2);

    const int m = 2;
    const int n = 1;
    const int k = 2;
    // forward pass
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
            thrust::raw_pointer_cast(&d_imdte_out1[0]), n
        )
    );
    // cudaDeviceSynchronize();  // don't need this since we're using cublas stream?
    /*
    two cuBLAS possible calls to add bias vector:
    */
    // using geam will require a beta value of 1 to scale input B properly
    //            important: remember that inputs are marked d_* to denote DEVICE matrix
    //              ex. don't forget to use d_b1 (not b1), otherwise CUDA error 700
    //            should probably use h_* for host stuff for best practice
    const int a_rows = 1;
    const int b_cols = 2;
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&d_imdte_out1[0]), a_rows,
            &geam_beta,
            thrust::raw_pointer_cast(&d_b1[0]), a_rows,
            thrust::raw_pointer_cast(&d_out1[0]), a_rows
        )
    );
    // using axpy will require the result to overwrite an input
    // CUBLAS_CHECK(
    //     cublasDaxpy(
    //         cublasH,
    //         2,
    //         &alpha,
    //         thrust::raw_pointer_cast(&d_imdte_out1[0]), 1,
    //         thrust::raw_pointer_cast(&d_b1[0]), 1
    //     )
    // );
    cudaStreamSynchronize(stream);  // sync here b/c don't know if thrust is synced with this stream
    thrust::transform(
        d_out1.begin(),
        d_out1.end(),
        d_out1.begin(),
        Sigmoid()
    );
    // layer 2
    CUBLAS_CHECK(
        cublasDgemm(
            cublasH, 
            CUBLAS_OP_N, 
            CUBLAS_OP_N, 
            n, k, m, 
            &alpha, 
            thrust::raw_pointer_cast(&d_out1[0]), n,
            thrust::raw_pointer_cast(&d_W2[0]), m, 
            &gemm_beta,
            thrust::raw_pointer_cast(&d_imdte_out2[0]), n
        )
    );
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&d_imdte_out2[0]), a_rows,
            &geam_beta,
            thrust::raw_pointer_cast(&d_b2[0]), a_rows,
            thrust::raw_pointer_cast(&d_out2[0]), a_rows
        )
    );
    cudaStreamSynchronize(stream);  // sync here b/c don't know if thrust is synced with this stream
    thrust::transform(
        d_out2.begin(),
        d_out2.end(),
        d_out2.begin(),
        Sigmoid()
    );

    // compute loss
    thrust::device_vector<d_type> d_target = {0.01, 0.99};
    thrust::device_vector<d_type> d_loss(2);
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&d_target[0]), a_rows,
            &negate_geam_beta,
            thrust::raw_pointer_cast(&d_out2[0]), a_rows,
            thrust::raw_pointer_cast(&d_loss[0]), a_rows
        )
    );
    cudaStreamSynchronize(stream);
    thrust::transform(
        d_loss.begin(),
        d_loss.end(),
        d_loss.begin(),
        MSE()
    );
    d_type total_error;
    CUBLAS_CHECK(
        cublasDasum(
            cublasH,
            2, 
            thrust::raw_pointer_cast(&d_loss[0]),
            1,
            &total_error
        )
    );
    
    // backward pass
    // ∂(error) / ∂(output) -- derivative of MSE
    thrust::device_vector<d_type> dEO(2);
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&d_out2[0]), a_rows,
            &negate_geam_beta,
            thrust::raw_pointer_cast(&d_target[0]), a_rows,
            thrust::raw_pointer_cast(&dEO[0]), a_rows
        )
    );
    // ∂(output) / ∂(net) -- derivative of activation function (sigmoid)
    thrust::device_vector<d_type> dON(2);
    thrust::device_vector<d_type> ones = {1, 1};
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&ones[0]), a_rows,
            &negate_geam_beta,
            thrust::raw_pointer_cast(&d_out2[0]), a_rows,
            thrust::raw_pointer_cast(&dON[0]), a_rows
        )
    );
    cudaStreamSynchronize(stream);
    thrust::transform(
        dON.begin(),
        dON.end(),
        d_out2.begin(),
        dON.begin(),
        thrust::multiplies<d_type>()
    );
    print_device_thrust<d_type>(1, 2, dON, 1);
    // ∂(net) / ∂(weight) -- derivative of linear transformation (WX + B)
    
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return EXIT_SUCCESS;
}