// bare-bones scalar autograd engine

#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <set>
#include <queue>
#include <vector>
#include <functional>

class Scalar {
public:
    // attributes (avoiding getter & setter mess)
    double _val;
    double _grad;
    std::set<Scalar> _prev;
    std::function<void()> _backward = []() {};
    // methods
    inline Scalar(double value): _val(value), _grad(0), _prev() {};
    inline Scalar(double value, const std::set<Scalar>& previous): _val(value), _grad(0), _prev(previous) {}
    ~Scalar() = default;
    // backward op.
    inline void backward() {
        // build out DAG topological sort containing ALL previous nodes
        std::vector<Scalar> top_sort;
        std::set<Scalar> visited;
        find_children(*this, top_sort, visited);
        // dz / dz = 1
        _grad = 1;
        // run ._backward() in reversed topological sort vector
        for (auto i = top_sort.rbegin(); i != top_sort.rend(); ++i) {
            i->_backward();
        }
    }
    // forward ops.
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
private:
    void find_children(const Scalar& scalar, std::vector<Scalar>& list, std::set<Scalar>& visited) {
        if (!visited.count(scalar)) {
            visited.insert(scalar);
            for (auto &i : scalar._prev) {
                find_children(i, list, visited);
            }
            list.push_back(scalar);
        }
    }
};

#endif  // SCALAR_HPP
