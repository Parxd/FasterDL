#include <unordered_set>

struct Value {
    double val;
    std::unordered_set<Value> prev;
    Value() = default;
    inline Value(double value): val(value), prev() {}
    // needs eq. operator + std::hash functor
    inline Value(double value, std::unordered_set<Value> previous): val(value), prev(previous) {}
    ~Value() = default;
    Value add(const Value&);
    Value mul(const Value&);
};
