#include <iomanip>

// Print Thrust host matrix
template <typename T>
void print_host_thrust(const int &m, const int &n, thrust::host_vector<T> V, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::fixed << std::setprecision(2) << V.data()[j * lda + i] << " ";
        }
        std::cout << "\n";
    }
}

// Print Thrust device matrix
template <typename T>
void print_device_thrust(const int &m, const int &n, thrust::device_vector<T> V, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::fixed << std::setprecision(2) << V.data()[j * lda + i] << " ";
        }
        std::cout << "\n";
    }
}