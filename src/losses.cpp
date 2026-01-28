#include "losses.hpp"
 
namespace lamp {
 
Tensor Loss::operator()(const Tensor& predictions, const Tensor& targets) {
    return forward(predictions, targets);
}
 
// =============================================================
// MSE Loss
// =============================================================
 
Tensor MSELoss::forward(const Tensor& predictions, const Tensor& targets) {
    // MSE = mean((predictions - targets)^2)
    Tensor diff = predictions - targets;
    Tensor squared = diff * diff;
    return squared.mean();
}
 
// =============================================================
// BCE Loss
// =============================================================
 
BCELoss::BCELoss(float eps) : eps_(eps) {}
 
Tensor BCELoss::forward(const Tensor& predictions, const Tensor& targets) {
    // BCE = -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions))
    // Clamp predictions for numerical stability
    Tensor pred_clamped = predictions.clamp(eps_, 1.0f - eps_);
 
    // Create tensors for 1.0
    std::vector<float> ones_data(predictions.data().size(), 1.0f);
    Tensor ones(ones_data, predictions.shape(), false);
 
    // log(predictions) and log(1 - predictions)
    Tensor log_pred = pred_clamped.log();
    Tensor log_one_minus_pred = (ones - pred_clamped).log();
 
    // targets * log(pred) + (1 - targets) * log(1 - pred)
    Tensor term1 = targets * log_pred;
    Tensor term2 = (ones - targets) * log_one_minus_pred;
 
    // Negative mean
    Tensor sum_terms = term1 + term2;
    Tensor loss = sum_terms.mean() * (-1.0f);
 
    return loss;
}
 
// =============================================================
// Cross Entropy Loss
// =============================================================
 
CrossEntropyLoss::CrossEntropyLoss(float eps) : eps_(eps) {}
 
Tensor CrossEntropyLoss::forward(const Tensor& predictions, const Tensor& targets) {
    // CrossEntropy = -mean(sum(targets * log(softmax(predictions))))
    // Apply softmax to get probabilities
    Tensor probs = predictions.softmax();
 
    // Clamp for numerical stability
    Tensor probs_clamped = probs.clamp(eps_, 1.0f - eps_);
 
    // log(softmax(predictions))
    Tensor log_probs = probs_clamped.log();
 
    // targets * log(probs)
    Tensor weighted = targets * log_probs;
 
    // -mean(sum(...))
    Tensor loss = weighted.sum() * (-1.0f / static_cast<float>(predictions.shape()[0]));
 
    return loss;
}
 
} // namespace lamp
 