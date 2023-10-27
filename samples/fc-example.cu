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
backward run of a one layer fully-connected
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

    thrust::device_vector<d_type> W = {0.10, 0.20, 0.30, 0.40, 0.50, 0.60}; // weights
    thrust::device_vector<d_type> X = {0.20, 0.40, 0.30};  // ONE input of 3 features
    thrust::device_vector<d_type> b = {0.50, 0.20};  // bias
    thrust::device_vector<d_type> Z(2);  // (WX + b)
    thrust::device_vector<d_type> Y(2);  // sig(WX + b)
    
    // m, n, k refer to (m * k) x (k * n) = (m * n)
    int m = 1;
    int k = 3;
    int n = 2;
    CUBLAS_CHECK(
        cublasDgemm(
            cublasH, 
            CUBLAS_OP_T, 
            CUBLAS_OP_N, 
            m, n, k,
            &alpha, 
            thrust::raw_pointer_cast(&X[0]), k,
            thrust::raw_pointer_cast(&W[0]), k, 
            &gemm_beta,
            thrust::raw_pointer_cast(&Z[0]), m
        )
    );
    // WTF--does cublas_op_t flag not explicitly transpose underlying matrix in memory???

    /*
    two cuBLAS possible calls to add bias vector:
    */
    // using geam will require a beta value of 1 to scale input B properly
    const int a_rows = 1;
    const int b_cols = 2;
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&Z[0]), a_rows,
            &geam_beta,
            thrust::raw_pointer_cast(&b[0]), a_rows,
            thrust::raw_pointer_cast(&Z[0]), a_rows
        )
    );
    // using axpy will require the result to overwrite an input
    // CUBLAS_CHECK(
    //     cublasDaxpy(
    //         cublasH,
    //         2,
    //         &alpha,
    //         thrust::raw_pointer_cast(&Z[0]), 1,
    //         thrust::raw_pointer_cast(&b[0]), 1
    //     )
    // );
    cudaStreamSynchronize(stream);  // sync here b/c don't know if thrust is synced with this stream
    thrust::transform(
        Z.begin(),
        Z.end(),
        Y.begin(),
        Sigmoid()
    );

    // compute loss
    thrust::device_vector<d_type> target = {0.96596, 0.45926};
    thrust::device_vector<d_type> L(2);
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&target[0]), a_rows,
            &negate_geam_beta,
            thrust::raw_pointer_cast(&Y[0]), a_rows,
            thrust::raw_pointer_cast(&L[0]), a_rows
        )
    );
    cudaStreamSynchronize(stream);
    thrust::transform(
        L.begin(),
        L.end(),
        L.begin(),
        MSE()
    );
    
    // --------------------------------------------------------------------------------
    // backward pass
    // full derivations: https://www.adityaagrawal.net/blog/deep_learning/bprop_fc
    /*
    ∂L / ∂Y is the derivative of error with respect to Y
    - it is the vector {y_1_out - y_1_target, y_2_out - y_2_target} in this case
    */
    thrust::device_vector<d_type> dLdY(2);
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&Y[0]), a_rows,
            &negate_geam_beta,
            thrust::raw_pointer_cast(&target[0]), a_rows,
            thrust::raw_pointer_cast(&dLdY[0]), a_rows
        )
    );
    /*
    ∂L / ∂Z is the derivative of error with respect to Z
    - it is equal to (∂L / ∂Y)(∂Y / ∂Z)
    - ∂Y / ∂Z = (Y)(1 - Y)
    - thus ∂L / ∂Z = (∂L / ∂Y)((Y)(1 - Y))
    */
    thrust::device_vector<d_type> ones(Y.size(), 1);
    thrust::device_vector<d_type> dYdZ(2);
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a_rows, b_cols,
            &alpha,
            thrust::raw_pointer_cast(&ones[0]), a_rows,
            &negate_geam_beta,
            thrust::raw_pointer_cast(&Y[0]), a_rows,
            thrust::raw_pointer_cast(&dYdZ[0]), a_rows
        )
    );
    cudaStreamSynchronize(stream);
    thrust::device_vector<d_type> dLdZ(2);
    thrust::transform(
        dYdZ.begin(),
        dYdZ.end(),
        Y.begin(),
        dYdZ.begin(),
        thrust::multiplies<d_type>()
    );
    thrust::transform(
        dLdY.begin(),
        dLdY.end(),
        dYdZ.begin(),
        dLdZ.begin(),
        thrust::multiplies<d_type>()
    );
    /*
    ∂L / ∂X is the derivative of error with respect to input X
    - it is equal to (∂L / ∂Z)(∂Z / ∂X)
    - (W^T)_rm x (∂L / ∂Z) = [(W^T)_cm x (∂L / ∂Z)]^T
    - the _rm specifies that (W^T) is the weight matrix in row-major
    - in other words, (W^T)_rm = (W)_cm and (W)_rm = (W^T)_cm
    - thus our equation is going to be (∂L / ∂Z) x (W^T)_cm
    */
    thrust::device_vector<d_type> dLdX(3);
    // (m * n) x (n * k)
    // for CUBLAS_OP_T, set dims assuming transposed version... (m * k) x (k * n) = (m * n)
    m = 1;
    k = 2;
    n = 3;
    CUBLAS_CHECK(
        cublasDgemm(
            cublasH, 
            CUBLAS_OP_N, 
            CUBLAS_OP_T, 
            m, n, k,
            &alpha, 
            thrust::raw_pointer_cast(&dLdZ[0]), m,
            thrust::raw_pointer_cast(&W[0]), n, 
            &gemm_beta,
            thrust::raw_pointer_cast(&dLdX[0]), m
        )
    );
    /*
    ∂L / ∂W is the derivative of error with respect to W
    - it is equal to (∂L / ∂Z)(∂Z / ∂W)
    - pretty similar to ∂L / ∂X, since Z = WX + b
    - row-major: (∂L / ∂Z) x (X^T)
    - column-major: (X^T) x (∂L / ∂Z)
    - column-major's (X^T) is row major's (X^T)^T, which is why we can switch order
      and get same (transposed) result

    - ∂L / ∂Z: (1 x 2)
    - X^T: (3 x 1)
    */
    thrust::device_vector<d_type> dLdW(6);
    m = 3;
    k = 1;
    n = 2;
    CUBLAS_CHECK(
        cublasDgemm(
            cublasH, 
            CUBLAS_OP_T, 
            CUBLAS_OP_N, 
            m, n, k,
            &alpha, 
            thrust::raw_pointer_cast(&X[0]), k,
            thrust::raw_pointer_cast(&dLdZ[0]), k, 
            &gemm_beta,
            thrust::raw_pointer_cast(&dLdW[0]), m
        )
    );
    // dLdW is how each weight affects loss
    // each column is how the previous layer's weights affects one neuron in this layer

    // using following image as an example,
    // https://www.adityaagrawal.net/blog/assets/deep_learning/bprop_fc_3.svg
    // [dL/dW_A1, dL/dW_B1]
    // [dL/dW_A2, dL/dW_B2]
    // [dL/dW_A3, dL/dW_B3]

    /*
    ∂L / ∂b is the derivative of error wrt. b
    - since the derivative of Z wrt. b is just 1, ∂L / ∂b = ∂L / ∂Z
    */
    thrust::device_vector<d_type> dLdB = dLdZ;
    // --------------------------------------------------------------------------------

    // updating weights (w/o learning rate)

    std::cout << "Old weights: \n";
    print_device_thrust<d_type>(3, 2, W, 3);

    m = 3;
    n = 2;
    CUBLAS_CHECK(
        cublasDgeam(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m, n,
            &alpha,
            thrust::raw_pointer_cast(&W[0]), m,
            &negate_geam_beta,
            thrust::raw_pointer_cast(&dLdW[0]), m,
            thrust::raw_pointer_cast(&W[0]), m
        )
    );

    std::cout << "New weights: \n";
    print_device_thrust<d_type>(3, 2, W, 3);
    
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}