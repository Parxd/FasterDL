#include <unordered_set>
#include "value.hpp"

Value Value::add(const Value &other) {
    return Value(val + other.val);
}
Value Value::mul(const Value &other) {
    return Value(val * other.val);
}
