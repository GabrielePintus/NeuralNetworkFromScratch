#include "tensor.hpp"

#include <iostream>
#include <string>

namespace {
void print_vector(const std::vector<float>& values, const std::string& label) {
    std::cout << label << " [";
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << values[i];
        if (i + 1 < values.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}
}


void print_progress(size_t step, size_t total, float loss) {
    const int bar_width = 40;
    float progress = float(step) / total;

    int pos = int(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0f) << "%  Loss: " << loss;
    std::cout.flush();
}


int main() {
    using namespace lamp;

    Tensor x({1.0f, 2.0f, 3.0f, 4.0f}, {4, 1}, false);
    Tensor y_true({3.0f, 5.0f, 7.0f, 9.0f}, {4, 1}, false);

    // Tensor w({0.0f}, {1, 1}, true);
    // Tensor b({0.0f}, {1}, true);
    Tensor w = Tensor::randn({1, 1}) * 0.02f;
    w.set_requires_grad(true);
    Tensor b({0.0f}, {1}, true);

    const float learning_rate = 0.05f;
    const size_t steps = 200;

    // print initial parameters
    std::cout << "Initial parameters:\n";
    print_vector(w.data(), "w:");
    print_vector(b.data(), "b:");

    for (size_t step = 0; step < steps; ++step) {
        Tensor preds = x.matmul(w).add_broadcast(b);
        Tensor diff = preds - y_true;
        Tensor loss = (diff * diff).mean();

        w.zero_grad();
        b.zero_grad();
        loss.backward();

        w[0] -= learning_rate * w.grad()[0];
        b[0] -= learning_rate * b.grad()[0];

        print_progress(step + 1, steps, loss.data()[0]);
    }

    std::cout << "\n"; // move to next line after bar finishes

    std::cout << "Final parameters:\n";
    print_vector(w.data(), "w:");
    print_vector(b.data(), "b:");

    return 0;
}