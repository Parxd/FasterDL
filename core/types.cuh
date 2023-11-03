#ifndef FAST_TYPES_CUH
#define FAST_TYPES_CUH

#include <unordered_map>

namespace fast {
    enum dTypes {
        F32,  // float -- float32
        F64,  // double -- float64
        I8,  // 8-bit int (signed)
        U8,  // 8-bit int (unsigned)
        I16,  // 16-bit int (signed)
        U16,  // 16-bit int (unsigned)
        I32,  // 32-bit int (signed)
        U32,  // 32-bit int (unsigned)
        I64,  // 64-bit int (signed)
        U64  // 64-bit int (unsigned)
    };
};

#endif // FAST_TYPES_CUH