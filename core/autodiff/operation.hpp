#ifndef OPERATION_HPP
#define OPERATION_HPP

class UnaryOperation {
public:
    virtual ~UnaryOperation() = default;
    virtual double backward(double grad) const = 0;
};

class BinaryOperation {
public:
    virtual ~BinaryOperation() = default;
    virtual double backwardA(double grad) const = 0;
    virtual double backwardB(double grad) const = 0;
};

#endif  // OPERATION_HPP