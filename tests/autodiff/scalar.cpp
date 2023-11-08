#include <iostream>
#include "../../core/autodiff/scalar.hpp"

int main(int argc, char* argv[]) {
    Scalar a(5.0231);
    Scalar b(1.3984);
    Scalar c(6.3919);
    auto d = (a + b) * c;

    std::cout<<d._val<<"\n";
    
    // d.backwards()
    // a._grad && b._grad && c._grad should be filled
}