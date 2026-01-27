#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace lamp;

void test_initialization() {
    std::cout << "Testing Initialization... ";
    Tensor t = Tensor::ones({2, 3});
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
    assert(t.data()[0] == 1.0f);
    std::cout << "OK\n";
}

void test_arithmetic() {
    std::cout << "Testing Arithmetic... ";
    Tensor a = Tensor::ones({2, 2});       // [1, 1; 1, 1]
    Tensor b = Tensor::ones({2, 2}) * 2.0f; // [2, 2; 2, 2]
    
    Tensor sum = a + b; // Should be [3, 3; 3, 3]
    assert(sum.data()[0] == 3.0f);
    
    Tensor diff = b - a; // Should be [1, 1; 1, 1]
    assert(diff.data()[0] == 1.0f);
    std::cout << "OK\n";
}

void test_matmul() {
    std::cout << "Testing Matmul... ";
    
    // Identity Matrix check
    // A = [1, 2]
    //     [3, 4]
    std::vector<float> data_a = {1, 2, 3, 4};
    Tensor A(data_a, {2, 2});
    
    // I = [1, 0]
    //     [0, 1]
    std::vector<float> data_i = {1, 0, 0, 1};
    Tensor I(data_i, {2, 2});
    
    Tensor C = A.matmul(I);
    
    // Result should still be A
    assert(C.data()[0] == 1);
    assert(C.data()[3] == 4);
    std::cout << "OK\n";
}

void test_chaining() {
    std::cout << "Testing Chaining (ReLU + Mean)... ";
    std::vector<float> data = {-1.0f, 0.0f, 1.0f, 2.0f};
    Tensor t(data, {4});
    
    // relu([-1, 0, 1, 2]) -> [0, 0, 1, 2]
    // mean([0, 0, 1, 2])  -> 3 / 4 = 0.75
    Tensor res = t.relu().mean();
    
    assert(res.data()[0] == 0.75f);
    std::cout << "OK\n";
}

void test_bigtensor_random() {
    std::cout << "Testing Large Tensor Random Initialization... " << std::endl;
    Tensor t = Tensor::randn({1000, 1000});
    assert(t.shape()[0] == 1000);
    assert(t.shape()[1] == 1000);

    std::cout << "Computing Mean... ";
    Tensor mean_t = t.mean();
    std::cout << "Mean: " << mean_t.data()[0] << std::endl;

    std::cout << "Computing Std Dev... ";
    Tensor std_t = t.std();
    std::cout << "Std Dev: " << std_t.data()[0] << std::endl;

    std::cout << "OK" << std::endl;
}

int main() {
    std::cout << "=== Running Lamp Tensor Tests ===\n";
    
    test_initialization();
    test_arithmetic();
    test_matmul();
    test_chaining();
    test_bigtensor_random();
    
    std::cout << "\nAll tests passed successfully!\n";
    
    return 0;
}