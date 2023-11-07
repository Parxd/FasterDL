#include <iostream>
#include "../../core/autodiff/graph/value.hpp"
#include "../../core/autodiff/operators/add.hpp"

int main(int argc, char* argv[]) {
    Value<double> val1(50.12905912);
    Value<double> val2(20.01232213);
    // val1 + val2
    Add<double>(val1, val2);
}
