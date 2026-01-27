#include "optimizer.hpp"

namespace lamp {

SGD::SGD(Module& module, float learning_rate)
    : learning_rate_(learning_rate) {
    for (const auto& entry : module.parameters()) {
        params_.push_back(entry.second);
    }
}

SGD::SGD(const std::vector<Tensor*>& params, float learning_rate)
    : params_(params),
      learning_rate_(learning_rate) {}

void SGD::step() {
    for (Tensor* param : params_) {
        if (!param) {
            continue;
        }
        const auto& grad = param->grad();
        if (grad.empty()) {
            continue;
        }
        for (size_t i = 0; i < grad.size(); ++i) {
            (*param)[i] -= learning_rate_ * grad[i];
        }
    }
}

void SGD::zero_grad() {
    for (Tensor* param : params_) {
        if (param) {
            param->zero_grad();
        }
    }
}

} // namespace lamp