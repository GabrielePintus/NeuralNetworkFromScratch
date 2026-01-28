#include "lamp/optim/optimizer.hpp"
#include <cmath>
 
namespace lamp {
namespace optim {
 
// =============================================================
// SGD Implementation
// =============================================================
 
SGD::SGD(nn::Module& module, float learning_rate)
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
 
// =============================================================
// Adam Implementation
// =============================================================
 
Adam::Adam(nn::Module& module, float learning_rate, float beta1, float beta2, float eps)
    : learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      t_(0) {
    for (const auto& entry : module.parameters()) {
        params_.push_back(entry.second);
    }
    init_state();
}
 
Adam::Adam(const std::vector<Tensor*>& params, float learning_rate, float beta1, float beta2, float eps)
    : params_(params),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      t_(0) {
    init_state();
}
 
void Adam::init_state() {
    for (Tensor* param : params_) {
        if (param) {
            size_t size = param->data().size();
            m_[param] = std::vector<float>(size, 0.0f);
            v_[param] = std::vector<float>(size, 0.0f);
        }
    }
}
 
void Adam::step() {
    t_++;  // Increment timestep
 
    for (Tensor* param : params_) {
        if (!param) {
            continue;
        }
        const auto& grad = param->grad();
        if (grad.empty()) {
            continue;
        }
 
        auto& m = m_[param];
        auto& v = v_[param];
 
        // Bias correction factors
        float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
        float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));
 
        for (size_t i = 0; i < grad.size(); ++i) {
            // Update first moment estimate
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * grad[i];
 
            // Update second moment estimate
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * grad[i] * grad[i];
 
            // Bias-corrected estimates
            float m_hat = m[i] / bias_correction1;
            float v_hat = v[i] / bias_correction2;
 
            // Update parameter
            (*param)[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}
 
void Adam::zero_grad() {
    for (Tensor* param : params_) {
        if (param) {
            param->zero_grad();
        }
    }
}
 
} // namespace optim
} // namespace lamp