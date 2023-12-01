#include <iostream>
#include "../../core/autodiff/value.hpp"

void test1() {
    std::cout << "\033[1;32m--- Test 1 ---\033[0m\n";
    auto a = Value(305);
    auto b = Value(123);
    auto c = Value(401);
}

int main(int argc, char* argv[]) {
    test1();
}
