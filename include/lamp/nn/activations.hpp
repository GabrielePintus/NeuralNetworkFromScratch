/**
 * @file activations.hpp
 * @brief Neural network activation function modules.
 * @author Lamp Project
 * @version 1.0
 */

#pragma once

#include "lamp/nn/module.hpp"

namespace lamp {
namespace nn {
 
/**
 * @brief ReLU activation module.
 *
 * Applies the Rectified Linear Unit function element-wise:
 * f(x) = max(0, x)
 */
class ReLU : public Module {
public:
    /**
     * @brief Applies ReLU activation.
     *
     * @param input Input tensor.
     * @return Tensor Output with ReLU applied element-wise.
     */
    Tensor forward(const Tensor& input) override;
};
 
/**
 * @brief Sigmoid activation module.
 *
 * Applies the Sigmoid function element-wise:
 * f(x) = 1 / (1 + exp(-x))
 */
class Sigmoid : public Module {
public:
    /**
     * @brief Applies Sigmoid activation.
     *
     * @param input Input tensor.
     * @return Tensor Output with Sigmoid applied element-wise.
     */
    Tensor forward(const Tensor& input) override;
};
 
/**
 * @brief Tanh activation module.
 *
 * Applies the hyperbolic tangent function element-wise:
 * f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 */
class Tanh : public Module {
public:
    /**
     * @brief Applies Tanh activation.
     *
     * @param input Input tensor.
     * @return Tensor Output with Tanh applied element-wise.
     */
    Tensor forward(const Tensor& input) override;
};
 
/**
 * @brief Softmax activation module.
 *
 * Applies Softmax normalization along the last dimension:
 * softmax(x_i) = exp(x_i) / sum(exp(x_j))
 */
class Softmax : public Module {
public:
    /**
     * @brief Applies Softmax activation.
     *
     * @param input Input tensor.
     * @return Tensor Normalized tensor where each row sums to 1.
     */
    Tensor forward(const Tensor& input) override;
};
 
/**
 * @brief Leaky ReLU activation module.
 *
 * Applies Leaky ReLU activation element-wise:
 * f(x) = x if x > 0, else alpha * x
 */
class LeakyReLU : public Module {
public:
    /**
     * @brief Construct Leaky ReLU with configurable negative slope.
     *
     * @param negative_slope Slope for negative values (default: 0.01).
     */
    explicit LeakyReLU(float negative_slope = 0.01f);

    /**
     * @brief Applies Leaky ReLU activation.
     *
     * @param input Input tensor.
     * @return Tensor Output with Leaky ReLU applied element-wise.
     */
    Tensor forward(const Tensor& input) override;

private:
    float negative_slope_;  ///< Slope for negative input values
};
 
} // namespace nn
} // namespace lamp