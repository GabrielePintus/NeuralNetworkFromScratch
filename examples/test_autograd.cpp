#include "linear.hpp"
#include "optimizer.hpp"
#include "sequential.hpp"

#include <iostream>
#include <memory>
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

    auto first = std::make_shared<Linear>(1, 32);
    auto second = std::make_shared<Linear>(32, 1);
    Sequential model({first, second});
    const float learning_rate = 0.001f;
    const size_t steps = 200;
    SGD optimizer(model, learning_rate);

    for (size_t step = 0; step < steps; ++step) {
        Tensor preds = model.forward(x);
        Tensor diff = preds - y_true;
        Tensor loss = (diff * diff).mean();

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        print_progress(step + 1, steps, loss.data()[0]);
    }

    auto& params = model.parameters();
    Tensor* weight = params.at("0.weight");
    Tensor* bias = params.at("0.bias");

    std::cout << "\nFinal parameters:\n";
    print_vector(weight->data(), "w:");
    print_vector(bias->data(), "b:");

    return 0;
}