// bare-bones scalar autograd engine modeled after karpathy/micrograd
// https://github.com/karpathy/micrograd

#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <set>
#include <functional>

class Scalar {
public:
    double _val;
    double _grad;
    std::set<Scalar> _prev;
    std::function<void()> _backward = []() {};

    inline Scalar(double value): _val(value), _grad(0), _prev() {};
    inline Scalar(double value, const std::set<Scalar>& previous): _val(value), _grad(0), _prev(previous) {}
    ~Scalar() = default;

    inline Scalar operator+(Scalar& other) {
        auto result = Scalar(_val + other._val, {*this, other});
        result._backward = [&]() {
            // x + y = z
            // dz / dx = 1
            // dz / dy = 1
            _grad += result._grad;
            other._grad += result._grad;
        };
        return result;
    }
    inline Scalar operator*(Scalar& other) {
        auto result = Scalar(_val * other._val, {*this, other});
        result._backward = [&]() {
            // xy = z
            // dz / dx = y
            // dz / dy = x
            _grad += other._val * result._grad;
            other._grad += _val * result._grad;
        };
        return result;
    }
    inline bool operator<(const Scalar& other) const {
        // need to overload for std::set
        return _val < other._val;
    }

    inline void backward() {
        // build out topological sort of graph containing ALL previous nodes
        // run ._backward() in reversed topological sort
        
    }
};

#endif
