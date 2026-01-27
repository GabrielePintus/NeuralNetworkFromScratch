#include "linear.hpp"

#include <iostream>
#include <string>
#include <vector>

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

    std::cout << "] " << int(progress * 100.0f)
              << "%  Loss: " << loss;

    std::cout.flush();
}

} // namespace

int main() {
    using namespace lamp;

    Tensor x({1.0f, 2.0f, 3.0f, 4.0f}, {4, 1}, false);
    Tensor y_true({3.0f, 5.0f, 7.0f, 9.0f}, {4, 1}, false);

    Linear layer(1, 1);
    auto& params = layer.parameters();
    Tensor* weight = params.at("weight");
    Tensor* bias = params.at("bias");

    const float learning_rate = 0.05f;
    const size_t steps = 200;

    for (size_t step = 0; step < steps; ++step) {
        Tensor preds = layer.forward(x);
        Tensor diff = preds - y_true;
        Tensor loss = (diff * diff).mean();

        weight->zero_grad();
        bias->zero_grad();
        loss.backward();

        (*weight)[0] -= learning_rate * weight->grad()[0];
        (*bias)[0] -= learning_rate * bias->grad()[0];

        print_progress(step + 1, steps, loss.data()[0]);
    }

    std::cout << "\nFinal parameters:\n";
    print_vector(weight->data(), "w:");
    print_vector(bias->data(), "b:");

    return 0;
}