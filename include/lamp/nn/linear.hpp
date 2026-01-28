/**
 * @file linear.hpp
 * @brief Fully connected (dense) linear layer implementation.
 * @author Gabriele Pintus
 * @version 1.0
 */

#pragma once
#include "lamp/nn/module.hpp"

namespace lamp {
namespace nn {

/**
 * @class Linear
 * @brief Fully connected linear transformation layer.
 *
 * Applies a linear transformation to incoming data: y = xW^T + b
 * where W is the weight matrix and b is the bias vector.
 *
 * The weights are initialized using Kaiming/He initialization,
 * which is suitable for ReLU-based networks.
 */
class Linear : public Module {
public:
    /**
     * @brief Construct a Linear layer.
     *
     * Initializes weights using Kaiming/He initialization and
     * bias with small random values.
     *
     * @param in_features Number of input features.
     * @param out_features Number of output features.
     * @param bias Whether to include a learnable bias term (default: true).
     */
    Linear(size_t in_features, size_t out_features, bool bias = true);

    /**
     * @brief Performs the forward pass: y = xW^T + b
     *
     * @param input Input tensor of shape (batch_size, in_features) or (in_features,).
     * @return Tensor Output tensor of shape (batch_size, out_features) or (out_features,).
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Access the weight matrix.
     *
     * @return const Tensor& Weight tensor of shape (out_features, in_features).
     */
    const Tensor& weight() const { return weight_; }

    /**
     * @brief Access the bias vector.
     *
     * @return const Tensor& Bias tensor of shape (out_features,).
     */
    const Tensor& bias() const { return bias_; }

private:
    size_t in_features_;   ///< Number of input features
    size_t out_features_;  ///< Number of output features
    bool use_bias_;        ///< Whether bias is enabled

    Tensor weight_;  ///< Weight matrix of shape (out_features, in_features)
    Tensor bias_;    ///< Bias vector of shape (out_features,)

    /**
     * @brief Initialize layer parameters using Kaiming/He initialization.
     */
    void init_parameters();
};

} // namespace nn
} // namespace lamp