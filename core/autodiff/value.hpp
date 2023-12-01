#include <set>

class Value {
public:
    Value() = default;
    Value(double value): data(value), gradient(0.0) {}
private:
    double data;
    double gradient;
    std::set<Value> children;
};
