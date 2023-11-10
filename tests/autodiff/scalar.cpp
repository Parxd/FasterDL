#include <iostream>
#include "../../core/autodiff/scalar.hpp"

void test1() {
    std::cout << "\n--- Test 1 ---\n";
    Scalar a(5.0231);
    Scalar b(1.3984);
    Scalar c(6.3919);
    auto d = (a + b) * c;
    d.backward();
    // a._grad && b._grad && c._grad should be non-zero
    std::cout << a._grad << "\n";
    std::cout << b._grad << "\n";
    std::cout << c._grad << "\n";
}

void test2() {
    std::cout << "\n--- Test 2 ---\n";
    Scalar a(103.1082);
    Scalar b(14.3013);
    Scalar c(439.329);
    Scalar d(285.1);
    Scalar e(203.2);
    Scalar f = (d * e + c) * a + b;
    f.backward();
    std::cout << e._grad << "\n";
}

void test3() {
    std::cout << "\n--- Test 3 ---\n";
    Scalar a(2.5391);
    Scalar b(3);
    Scalar c = a ^ b;
    c.backward();
    std::cout << a._grad << "\n";
    std::cout << b._grad << "\n";
}

int main(int argc, char* argv[]) {
    // test1();
    // test2();
    test3();
}
