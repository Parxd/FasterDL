# "essentials" for cuBLAS programming

- handle & stream
- `m`, `n`, `k`
- `ld{a/b/c}`
- `alpha`, `beta` values
- host data
- device **pointers**
- device memory allocation
- data copy to device
- **computation**
- data copy to host
- stream sync

## handle & stream
```c++
cublasHandle_t handle = NULL;
cudaStream_t stream = NULL;

cublasCreate(&handle);
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
cublasSetStream(handle, stream));
```

## `m`, `n`, `k`
for gemm, the variables represent the following:
- `m` -- rows in `A` & `C`
- `n` -- cols in `B` & `C`
- `k` -- cols in `A` & rows in `B`

#### shapes
- `A` of `m * k`
- `B` of `k * n`
- `A x B` = `C` of `m * n`

```c++
// matrices
const int m = 2;
const int n = 3;
const int k = 3;

const int lda;
const int ldb;
const int ldc;
```

## `lda`, `ldb`, `ldc`, etc.:

leading dimensions, equal to number of rows in column-major systems

essentially number of elements that are processed per "time" when data is laid sequentially

in column-major like cuBLAS, its the number of elements taken and placed in a column vertically

for ex:
- 2x3 matrix as vector (rows = 2, cols = 3):

    **| 1 | 2 | 3 |**

    **| 4 | 5 | 6 |**

- column-major -- [1, 4, 2, 5, 3, 6]
- row-major -- [1, 2, 3, 4, 5, 6]
- *FOR CM SYSTEMS: `lda` = **2**, and processes the vector above `lda` elements at a time and places them in a column*
- *FOR RM SYSTEMS: `lda` = **3**, and processes the vector above `lda` elements at a time and places them in a row*

More: https://www.adityaagrawal.net/blog/deep_learning/row_column_major

---

*in summary:* suppose we have some vector `A` = {9, 1, 2, 6, 3, 8} & `r` = 2 & `c` = 3 in memory

how this is represented as a matrix can vary depending on the system

- if we pass `A` to *CM-based*:

    **| 9 | 2 | 3 |**

    **| 1 | 6 | 8 |**

- if we pass `A` to *RM-based*:

    **| 9 | 1 | 2 |**

    **| 6 | 3 | 8 |**

suppose we take `A`, do stuff w/ it in CM-based, and want to convert it to RM-based:
- switch r & c
- `A` becomes the *transpose* of itself in CM-based, now in RM-based

**NIMPORTANT**: A x B = (B<sup>T</sup> x A<sup>T</sup>)<sup>T</sup>

## `alpha`, `beta` values
```c++
float alpha = 1.0;
float beta = 0.0;
```
- MUST BE IN SAME D_TYPE as matrix data
- passed by ref. to most of cublas API
- `alpha` scales one of the input matrices `A` or `B`; results in the same `C` matrix regardless
- `beta` scales the output matrix `C` before `A x B` is computed into it
    - **0.0**: existing values are ignored
    - **1.0**: result of operation added to existing values
    - **else**: scales existing values before adding result of operation

## host data
```c++
const std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
const std::vector<double> B = {5.0, 6.0, 7.0, 8.0};
std::vector<double> C(m * n, 0);
```
**OR**
```c++
double A[m * k];
double B[k * n];
double C[m * n];
```
