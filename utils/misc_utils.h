#include <thrust/random.h>

// Generate random normalized value 0-1
template <typename T> T random_norm() {
    static thrust::default_random_engine rng;
    static thrust::uniform_real_distribution<T> dist(0, 1);
    return dist(rng);
}

