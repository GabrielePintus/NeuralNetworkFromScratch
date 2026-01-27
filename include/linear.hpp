#pragma once
#include "module.hpp"

namespace lamp {

class Linear : public Module {
public:
    /**
     * @brief Fully connected linear layer: y = xW^T + b
     * @param in_features Input dimension
     * @param out_features Output dimension
     * @param bias Whether to include bias term
     */
    Linear(size_t in_features, size_t out_features, bool bias = true);
    
    Tensor forward(const Tensor& input) override;
    
    const Tensor& weight() const { return weight_; }
    const Tensor& bias() const { return bias_; }

private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    
    Tensor weight_;  // Shape: (out_features, in_features)
    Tensor bias_;    // Shape: (out_features,)
    
    void init_parameters();
};

} // namespace lamp