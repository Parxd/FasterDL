#include <cstdlib>
#include <vector>
#include <unordered_map>
#include "node.hpp"

template <typename DataType>
class Value;

template <typename DataType>
struct ValueHash {
    size_t operator()(const Value<DataType>& val) const {
        return (
            (std::hash<DataType>()(val._out)) ^ 
            (std::hash<DataType>()(val._grad) << 1) >> 1
        );
    }
};

template <typename DataType>
class Value: public Node {
public:
    DataType _out;
    DataType _grad = 0;
    std::unordered_map<Node*, DataType, ValueHash<DataType>> _grads;
    std::vector<Node*> _in_nodes;
    std::vector<Node*> _out_nodes;

    Value() = default;
    inline Value(DataType val): _out(static_cast<DataType>(val)) {}
    ~Value() = default;
    inline void forward() override {
        
    }
    inline void backward() override {
        
    }
    inline bool operator==(const Value& other) const {
        return (
            _out == other._out &&
            _grad == other._grad &&
            _grads == other._grads &&
            _in_nodes == other._in_nodes &&
            _out_nodes == other._out_nodes
        );
    }
};
