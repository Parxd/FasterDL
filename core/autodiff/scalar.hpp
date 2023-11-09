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
    ~Scalar() = default;
    // backward op.
    inline void backward() {
        // build out DAG topological sort containing ALL previous nodes using bfs traversal
        std::set<Scalar> visited;
        std::vector<Scalar> top_sort;
        std::queue<Scalar> bfs_queue;
        find_children_bfs(*this, top_sort, visited, bfs_queue);
        // dz / dz = 1
        _grad = 1;
        // run ._backward() in topological sort vector
        for (auto &i : top_sort) {
            // std::cout << i._val << "\n";
            i._backward();
        }
        // ...OR using dfs traversal
        // find_children_dfs(*this, top_sort, visited);
        // for (auto i = top_sort.rbegin(); i != top_sort.rend(); ++i) {
        //     std::cout << i->_val << "\n";
        //     i->_backward();
        // }
    }
    // forward ops.
    inline Scalar operator+(Scalar& other) {
        auto result = Scalar(_val + other._val, {*this, other});
        result._backward = [&]() {
            _grad += result._grad;
            other._grad += result._grad;
        };
        return result;
    }
    inline Scalar operator*(Scalar& other) {
        auto result = Scalar(_val * other._val, {*this, other});
        result._backward = [&]() {
            _grad += other._val * result._grad;
            other._grad += _val * result._grad;
        };
        return result;
    }
    // need to overload for std::set
    inline bool operator<(const Scalar& other) const {
        return _val < other._val;
    }
private:
    inline Scalar(double value, const std::set<Scalar>& previous): _val(value), _grad(0), _prev(previous) {}
    void find_children_bfs(const Scalar& root, std::vector<Scalar>& sort, std::set<Scalar>& visited, std::queue<Scalar>& queue) {
        visited.insert(root);
        queue.push(root);
        while (queue.size()) {
            auto v = queue.front();
            sort.push_back(v);
            queue.pop();
            for (auto &i : v._prev) {
                if (!visited.count(i)) {
                    queue.push(i);
                    visited.insert(i);
                }
            }
        }
    }
    void find_children_dfs(const Scalar& scalar, std::vector<Scalar>& list, std::set<Scalar>& visited) {
        if (!visited.count(scalar)) {
            visited.insert(scalar);
            for (auto &i : scalar._prev) {
                find_children_dfs(i, list, visited);
            }
            list.push_back(scalar);
        }
    }
};

#endif  // SCALAR_HPP
