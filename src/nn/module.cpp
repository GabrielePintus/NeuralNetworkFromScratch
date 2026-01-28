#include "lamp/nn/module.hpp"

namespace lamp {
namespace nn {

Module::~Module() = default;

Tensor Module::operator()(const Tensor& input) {
    return forward(input);
}

void Module::train() {
    training_ = true;
}

void Module::eval() {
    training_ = false;
}

bool Module::is_training() const {
    return training_;
}

void Module::zero_grad() {
    for (auto& entry : parameters_) {
        if (entry.second) {
            entry.second->zero_grad();
        }
    }
}

void Module::register_parameter(const std::string& name, Tensor& param) {
    parameters_[name] = &param;
}

std::unordered_map<std::string, Tensor*>& Module::parameters() {
    return parameters_;
}

} // namespace nn
} // namespace lamp