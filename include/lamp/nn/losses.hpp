#pragma once
 
#include "lamp/core/tensor.hpp"
 
namespace lamp {
namespace nn {
 
/**
 * @brief Base class for all loss functions.
 */
class Loss {
public:
    virtual ~Loss() = default;
    virtual Tensor forward(const Tensor& predictions, const Tensor& targets) = 0;
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
    explicit BCELoss(float eps = 1e-7f);
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;
private:
    float eps_;  // Small value for numerical stability
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
    explicit CrossEntropyLoss(float eps = 1e-7f);
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;
private:
    float eps_;
};
 
} // namespace nn
} // namespace lamp