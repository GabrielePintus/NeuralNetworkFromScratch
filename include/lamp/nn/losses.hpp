/**
 * @file losses.hpp
 * @brief Loss function implementations for neural network training.
 * @author Gabriele Pintus
 * @version 1.0
 */

#pragma once

#include "lamp/core/tensor.hpp"

namespace lamp {
namespace nn {
 
/**
 * @brief Base class for all loss functions.
 */
class Loss {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Loss() = default;

    /**
     * @brief Computes the loss between predictions and targets.
     *
     * @param predictions The predicted values from the model.
     * @param targets The ground truth values.
     * @return Tensor The computed loss value.
     */
    virtual Tensor forward(const Tensor& predictions, const Tensor& targets) = 0;

    /**
     * @brief Convenience operator for calling forward().
     *
     * @param predictions The predicted values.
     * @param targets The ground truth values.
     * @return Tensor The computed loss value.
     */
    Tensor operator()(const Tensor& predictions, const Tensor& targets);
};
 
/**
 * @brief Mean Squared Error Loss.
 *
 * Computes: mean((predictions - targets)^2)
 * Used for regression tasks.
 */
class MSELoss : public Loss {
public:
    /**
     * @brief Computes mean squared error loss.
     *
     * @param predictions Predicted values.
     * @param targets Ground truth values.
     * @return Tensor Scalar loss value.
     */
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;
};
 
/**
 * @brief Binary Cross Entropy Loss.
 *
 * Computes: -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions))
 * Used for binary classification with sigmoid output.
 */
class BCELoss : public Loss {
public:
    /**
     * @brief Construct BCE loss with numerical stability epsilon.
     *
     * @param eps Small value added for numerical stability (default: 1e-7).
     */
    explicit BCELoss(float eps = 1e-7f);

    /**
     * @brief Computes binary cross entropy loss.
     *
     * @param predictions Predicted probabilities (should be in [0, 1]).
     * @param targets Ground truth binary labels (0 or 1).
     * @return Tensor Scalar loss value.
     */
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;

private:
    float eps_;  ///< Small value for numerical stability
};
 
/**
 * @brief Cross Entropy Loss with Softmax.
 *
 * Computes: -mean(sum(targets * log(softmax(predictions))))
 * Used for multi-class classification.
 *
 * Note: Expects raw logits, applies softmax internally.
 */
class CrossEntropyLoss : public Loss {
public:
    /**
     * @brief Construct cross entropy loss with numerical stability epsilon.
     *
     * @param eps Small value added for numerical stability (default: 1e-7).
     */
    explicit CrossEntropyLoss(float eps = 1e-7f);

    /**
     * @brief Computes cross entropy loss with softmax.
     *
     * @param predictions Raw logits (pre-softmax values).
     * @param targets One-hot encoded ground truth labels.
     * @return Tensor Scalar loss value.
     */
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;

private:
    float eps_;  ///< Small value for numerical stability
};
 
} // namespace nn
} // namespace lamp