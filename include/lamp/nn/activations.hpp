#pragma once
 
#include "lamp/nn/module.hpp"
 
namespace lamp {
namespace nn {
 
/**
 * @brief ReLU activation module.
 *
 * f(x) = max(0, x)
 */
class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
};
 
/**
 * @brief Sigmoid activation module.
 *
 * f(x) = 1 / (1 + exp(-x))
 */
class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& input) override;
};
 
/**
 * @brief Tanh activation module.
 *
 * f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 */
class Tanh : public Module {
public:
    Tensor forward(const Tensor& input) override;
};
 
/**
 * @brief Softmax activation module.
 *
 * Applies softmax along the last dimension.
 * softmax(x_i) = exp(x_i) / sum(exp(x_j))
 */
class Softmax : public Module {
public:
    Tensor forward(const Tensor& input) override;
};
 
/**
 * @brief Leaky ReLU activation module.
 *
 * f(x) = x if x > 0, else alpha * x
 */
class LeakyReLU : public Module {
public:
    explicit LeakyReLU(float negative_slope = 0.01f);
    Tensor forward(const Tensor& input) override;
private:
    float negative_slope_;
};
 
} // namespace nn
} // namespace lamp