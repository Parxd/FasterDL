#ifndef FAST_TYPES
#define FAST_TYPES

#include <unordered_map>

typedef enum dataType_t {
    F16,  // half -- float16
    F32,  // float -- float32
    F64,  // double -- float64
    I4,  // 4-bit int (signed)
    U4,  // 4-bit int (unsigned)
    I8,  // 8-bit int (signed)
    U8,  // 8-bit int (unsigned)
    I16,  // 16-bit int (signed)
    U16,  // 16-bit int (unsigned)
    I32,  // 32-bit int (signed)
    U32,  // 32-bit int (unsigned)
    I64,  // 64-bit int (signed)
    U64  // 64-bit int (unsigned)
} dataType;

F16 s;

#endif // FAST_TYPES