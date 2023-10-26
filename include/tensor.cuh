#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace Fast {
    class Tensor {
    public:
        Tensor();
        Tensor(int);
        Tensor(int, int);
        Tensor(std::vector<cudaDataType>);
        Tensor(thrust::host_vector<cudaDataType>);
        Tensor(thrust::device_vector<cudaDataType>);
        ~Tensor();

    private:
        size_t rows;
        size_t cols;
        size_t leading;
        bool rm;
        bool cm;
        thrust::device_vector<cudaDataType> data;
    };
}

#endif TENSOR_CUH