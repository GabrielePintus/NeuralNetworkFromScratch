#include "lamp/nn/activations.hpp"
 
namespace lamp {
namespace nn {

// =============================================================
// ReLU
// =============================================================
 
Tensor ReLU::forward(const Tensor& input) {
    return input.relu();
}
 
// =============================================================
// Sigmoid
// =============================================================
 
Tensor Sigmoid::forward(const Tensor& input) {
    return input.sigmoid();
}
 
// =============================================================
// Tanh
// =============================================================
 
Tensor Tanh::forward(const Tensor& input) {
    return input.tanh();
}
 
// =============================================================
// Softmax
// =============================================================
 
Tensor Softmax::forward(const Tensor& input) {
    return input.softmax();
}
 
// =============================================================
// LeakyReLU
// =============================================================
 
LeakyReLU::LeakyReLU(float negative_slope) : negative_slope_(negative_slope) {}
 
Tensor LeakyReLU::forward(const Tensor& input) {
    return input.leaky_relu(negative_slope_);
}
 
} // namespace nn
} // namespace lamp