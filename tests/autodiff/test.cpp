#include <iostream>
#include "../../core/autodiff/value.hpp"

int main(int argc, char* argv[]) {
    Value a(3.0);
    Value b(-2.0);
    Value c(12.0);
    Value d = a.mul(b).add(c);
    std::cout << d.val << "\n\n";
    for (auto & i : d.prev) {
        std::cout << i.val << "\n";
    }
}
