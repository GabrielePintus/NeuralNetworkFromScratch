#include "linear.hpp"
#include <cmath>

namespace lamp {

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(bias),
      weight_(Tensor::zeros({out_features_, in_features_})),
      bias_(Tensor::zeros({out_features_})) {
    
    init_parameters();
    weight_.set_requires_grad(true);
    if (use_bias_) {
        bias_.set_requires_grad(true);
    }
    
    // Register parameters
    register_parameter("weight", weight_);
    if (use_bias_) {
        register_parameter("bias", bias_);
    }
}

void Linear::init_parameters() {
    // Kaiming/He initialization for ReLU networks
    // std = sqrt(2 / in_features)
    float std = std::sqrt(2.0f / in_features_);
    weight_ = Tensor::randn({out_features_, in_features_}) * std;
    
    if (use_bias_) {
        // Initialize bias to small random values
        bias_ = Tensor::uniform({out_features_}, -0.01f, 0.01f);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // Input shape: (batch_size, in_features) or (in_features,)
    // Weight shape: (out_features, in_features)
    // Output shape: (batch_size, out_features) or (out_features,)
    
    // Compute: input @ weight^T
    Tensor output = input.matmul(weight_.transpose());
    
    // Add bias if enabled
    if (use_bias_) {
        // Broadcast bias across batch dimension
        output = output + bias_;
    }
    
    return output;
}

} // namespace lamp