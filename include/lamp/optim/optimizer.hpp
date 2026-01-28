/**
 * @file optimizer.hpp
 * @brief Optimization algorithms for neural network training.
 * @author Gabriele Pintus
 * @version 1.0
 */

#pragma once

#include "lamp/nn/module.hpp"
#include <vector>
#include <unordered_map>

namespace lamp {
namespace optim {
 
/**
 * @brief Stochastic Gradient Descent optimizer.
 *
 * Updates parameters using: θ = θ - lr * ∇θ
 * where lr is the learning rate and ∇θ are the gradients.
 */
class SGD {
public:
    /**
     * @brief Construct SGD optimizer from a module's parameters.
     *
     * @param module The module whose parameters to optimize.
     * @param learning_rate Step size for parameter updates.
     */
    SGD(nn::Module& module, float learning_rate);

    /**
     * @brief Construct SGD optimizer from a list of parameters.
     *
     * @param params Vector of pointers to tensors to optimize.
     * @param learning_rate Step size for parameter updates.
     */
    SGD(const std::vector<Tensor*>& params, float learning_rate);

    /**
     * @brief Perform one optimization step.
     *
     * Updates all parameters based on their gradients.
     */
    void step();

    /**
     * @brief Zero out all parameter gradients.
     */
    void zero_grad();

private:
    std::vector<Tensor*> params_;  ///< Parameters to optimize
    float learning_rate_;          ///< Learning rate
};
 
 
/**
 * @brief Adam optimizer (Adaptive Moment Estimation).
 *
 * Updates parameters using first and second moment estimates of gradients.
 * Algorithm:
 *   m_t = β1 * m_{t-1} + (1 - β1) * g_t
 *   v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
 *   m̂_t = m_t / (1 - β1^t)
 *   v̂_t = v_t / (1 - β2^t)
 *   θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
 */
class Adam {
public:
    /**
     * @brief Construct Adam optimizer from a module's parameters.
     * @param module The module whose parameters to optimize.
     * @param learning_rate Learning rate (default: 0.001).
     * @param beta1 Exponential decay rate for first moment (default: 0.9).
     * @param beta2 Exponential decay rate for second moment (default: 0.999).
     * @param eps Small constant for numerical stability (default: 1e-8).
     */
    Adam(nn::Module& module, float learning_rate = 0.001f, float beta1 = 0.9f,
         float beta2 = 0.999f, float eps = 1e-8f);
 
    /**
     * @brief Construct Adam optimizer from a list of parameters.
     */
    Adam(const std::vector<Tensor*>& params, float learning_rate = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
 
    /**
     * @brief Perform one optimization step.
     */
    void step();
 
    /**
     * @brief Zero out all parameter gradients.
     */
    void zero_grad();
 
private:
    std::vector<Tensor*> params_;
    float learning_rate_;
    float beta1_;
    float beta2_;
    float eps_;
    size_t t_;  // Timestep counter
 
    // First moment (mean) estimates for each parameter
    std::unordered_map<Tensor*, std::vector<float>> m_;
 
    // Second moment (variance) estimates for each parameter
    std::unordered_map<Tensor*, std::vector<float>> v_;
 
    void init_state();
 
};


} // namespace optim
} // namespace lamp