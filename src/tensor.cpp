/**
 * @file tensor.cpp
 * @brief Implementation of the Tensor class.
 */

#include "tensor.hpp"
#include <random>
#include <iomanip>
#include <unordered_set>

namespace lamp {

// =============================================================
// Constructor & Validation
// =============================================================

Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape, bool requires_grad)
    : node_(std::make_shared<TensorNode>()) {
    node_->data = std::move(data);
    node_->shape = std::move(shape);
    node_->requires_grad = requires_grad;
    validate();
}

void Tensor::validate() const {
    size_t required_size = std::accumulate(node_->shape.begin(), node_->shape.end(), 1, std::multiplies<size_t>());
    if (node_->data.size() != required_size) {
        throw std::invalid_argument("Tensor Error: Data size (" + std::to_string(node_->data.size()) +
                                  ") does not match shape capacity (" + std::to_string(required_size) + ")");
    }
}

void Tensor::ensure_grad(bool zero) {
    if (node_->grad.empty()) {
        node_->grad.assign(node_->data.size(), 0.0f);
        return;
    }
    if (zero) {
        std::fill(node_->grad.begin(), node_->grad.end(), 0.0f);
    }
}

// =============================================================
// Private Helpers
// =============================================================

Tensor Tensor::map(std::function<float(float)> op) const {
    std::vector<float> result = node_->data;
    std::transform(result.begin(), result.end(), result.begin(), op);
    return Tensor(std::move(result), node_->shape, node_->requires_grad);
}

Tensor Tensor::zip(const Tensor& other, std::function<float(float, float)> op) const {
    if (node_->shape != other.node_->shape) {
        throw std::invalid_argument("Tensor Error: Shape mismatch in element-wise operation.");
    }
    std::vector<float> result(node_->data.size());
    std::transform(node_->data.begin(), node_->data.end(), other.node_->data.begin(), result.begin(), op);
    return Tensor(std::move(result), node_->shape, node_->requires_grad || other.node_->requires_grad);
}

// =============================================================
// Factory Methods
// =============================================================

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    return Tensor(std::vector<float>(size, 0.0f), shape);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    auto t = zeros(shape);
    std::fill(t.node_->data.begin(), t.node_->data.end(), 1.0f);
    return t;
}

Tensor Tensor::randn(const std::vector<size_t>& shape) {
    auto t = zeros(shape);
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : t.node_->data) x = dist(gen);
    return t;
}

Tensor Tensor::uniform(const std::vector<size_t>& shape, float low, float high) {
    auto t = zeros(shape);
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& x : t.node_->data) x = dist(gen);
    return t;
}

// =============================================================
// Operators
// =============================================================

// Tensor Tensor::operator+(const Tensor& other) const {
//     // If shapes match exactly, use element-wise addition
//     if (node_->shape == other.node_->shape) {
//         auto out = zip(other, std::plus<float>());
//         if (out.node_->requires_grad) {
//             auto out_node = out.node_;
//             auto left = node_;
//             auto right = other.node_;
//             out_node->parents = {left, right};
//             out_node->backward = [out_node, left, right]() {
//                 if (left->requires_grad) {
//                     if (left->grad.empty()) {
//                         left->grad.assign(left->data.size(), 0.0f);
//                     }
//                     for (size_t i = 0; i < left->grad.size(); ++i) {
//                         left->grad[i] += out_node->grad[i];
//                     }
//                 }
//                 if (right->requires_grad) {
//                     if (right->grad.empty()) {
//                         right->grad.assign(right->data.size(), 0.0f);
//                     }
//                     for (size_t i = 0; i < right->grad.size(); ++i) {
//                         right->grad[i] += out_node->grad[i];
//                     }
//                 }
//             };
//         }
//         return out;
//     }
    
//     // Handle broadcasting: (M, N) + (N,)
//     if (node_->shape.size() == 2 && other.node_->shape.size() == 1) {
//         if (node_->shape[1] == other.node_->shape[0]) {
//             return add_broadcast(other);
//         }
//     }
    
//     // Handle broadcasting: (N,) + (M, N)
//     if (node_->shape.size() == 1 && other.node_->shape.size() == 2) {
//         if (node_->shape[0] == other.node_->shape[1]) {
//             return other.add_broadcast(*this);
//         }
//     }
    
//     throw std::invalid_argument(
//         "Tensor Error: Incompatible shapes for addition. Broadcasting not supported for these shapes."
//     );
// }
Tensor Tensor::operator+(const Tensor& other) const {
    // If shapes match exactly, use element-wise addition
    if (node_->shape == other.node_->shape) {
        auto out = zip(other, std::plus<float>());

        if (out.node_->requires_grad) {
            // Keep owning shared_ptr only outside the lambda
            auto out_node_sp = out.node_;
            TensorNode* out_node = out_node_sp.get();   // raw pointer to avoid cycles

            auto left  = node_;
            auto right = other.node_;

            out_node_sp->parents = { left, right };

            out_node_sp->backward = [out_node, left, right]() {
                // NOTE: out_node is a raw ptr, valid as long as this backward runs
                // (and it will, because the node owns the backward function)

                if (left->requires_grad) {
                    if (left->grad.empty()) {
                        left->grad.assign(left->data.size(), 0.0f);
                    }
                    for (size_t i = 0; i < left->grad.size(); ++i) {
                        left->grad[i] += out_node->grad[i];   // dL/dleft += dL/dout
                    }
                }

                if (right->requires_grad) {
                    if (right->grad.empty()) {
                        right->grad.assign(right->data.size(), 0.0f);
                    }
                    for (size_t i = 0; i < right->grad.size(); ++i) {
                        right->grad[i] += out_node->grad[i];  // dL/dright += dL/dout
                    }
                }
            };
        }

        return out;
    }

    // Handle broadcasting: (M, N) + (N,)
    if (node_->shape.size() == 2 && other.node_->shape.size() == 1) {
        if (node_->shape[1] == other.node_->shape[0]) {
            return add_broadcast(other);
        }
    }

    // Handle broadcasting: (N,) + (M, N)
    if (node_->shape.size() == 1 && other.node_->shape.size() == 2) {
        if (node_->shape[0] == other.node_->shape[1]) {
            return other.add_broadcast(*this);
        }
    }

    throw std::invalid_argument(
        "Tensor Error: Incompatible shapes for addition. Broadcasting not supported for these shapes."
    );
}
Tensor Tensor::add_broadcast(const Tensor& vec) const {
    if (node_->shape.size() != 2) {
        throw std::invalid_argument("add_broadcast: requires 2D tensor");
    }
    if (vec.node_->shape.size() != 1) {
        throw std::invalid_argument("add_broadcast: requires 1D vector");
    }
    if (node_->shape[1] != vec.node_->shape[0]) {
        throw std::invalid_argument("add_broadcast: vector size must match tensor columns");
    }
    
    size_t rows = node_->shape[0];
    size_t cols = node_->shape[1];
    std::vector<float> result = node_->data;
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i * cols + j] += vec.node_->data[j];
        }
    }
    
    Tensor out(std::move(result), node_->shape, node_->requires_grad || vec.node_->requires_grad);
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto left = node_;
        auto right = vec.node_;
        out_node->parents = {left, right};
        out_node->backward = [out_node, left, right, rows, cols]() {
            if (left->requires_grad) {
                if (left->grad.empty()) {
                    left->grad.assign(left->data.size(), 0.0f);
                }
                for (size_t i = 0; i < left->grad.size(); ++i) {
                    left->grad[i] += out_node->grad[i];
                }
            }
            if (right->requires_grad) {
                if (right->grad.empty()) {
                    right->grad.assign(right->data.size(), 0.0f);
                }
                for (size_t j = 0; j < cols; ++j) {
                    float accum = 0.0f;
                    for (size_t i = 0; i < rows; ++i) {
                        accum += out_node->grad[i * cols + j];
                    }
                    right->grad[j] += accum;
                }
            }
        };
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    auto out = zip(other, std::minus<float>());

    if (out.node_->requires_grad) {
        // Keep shared_ptr only outside the lambda
        auto out_node_sp = out.node_;
        TensorNode* out_node = out_node_sp.get();   // raw pointer (no ownership)

        auto left  = node_;
        auto right = other.node_;

        out_node_sp->parents = { left, right };

        out_node_sp->backward = [out_node, left, right]() {
            if (left->requires_grad) {
                if (left->grad.empty()) {
                    left->grad.assign(left->data.size(), 0.0f);
                }
                for (size_t i = 0; i < left->grad.size(); ++i) {
                    left->grad[i] += out_node->grad[i];   // dL/dleft = dL/dout
                }
            }

            if (right->requires_grad) {
                if (right->grad.empty()) {
                    right->grad.assign(right->data.size(), 0.0f);
                }
                for (size_t i = 0; i < right->grad.size(); ++i) {
                    right->grad[i] -= out_node->grad[i];  // dL/dright = -dL/dout
                }
            }
        };
    }

    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    auto out = zip(other, std::multiplies<float>());

    if (out.node_->requires_grad) {
        // Hold shared_ptr only outside the lambda
        auto out_node_sp = out.node_;
        TensorNode* out_node = out_node_sp.get();   // raw pointer (no ownership)

        auto left  = node_;
        auto right = other.node_;

        out_node_sp->parents = { left, right };

        out_node_sp->backward = [out_node, left, right]() {
            if (left->requires_grad) {
                if (left->grad.empty()) {
                    left->grad.assign(left->data.size(), 0.0f);
                }
                for (size_t i = 0; i < left->grad.size(); ++i) {
                    left->grad[i] += out_node->grad[i] * right->data[i];
                }
            }

            if (right->requires_grad) {
                if (right->grad.empty()) {
                    right->grad.assign(right->data.size(), 0.0f);
                }
                for (size_t i = 0; i < right->grad.size(); ++i) {
                    right->grad[i] += out_node->grad[i] * left->data[i];
                }
            }
        };
    }

    return out;
}

Tensor Tensor::operator/(const Tensor& other) const {
    auto out = zip(other, std::divides<float>());

    if (out.node_->requires_grad) {
        // Hold shared_ptr only outside the lambda
        auto out_node_sp = out.node_;
        TensorNode* out_node = out_node_sp.get();   // raw pointer (no ownership)

        auto left  = node_;
        auto right = other.node_;

        out_node_sp->parents = { left, right };

        out_node_sp->backward = [out_node, left, right]() {
            if (left->requires_grad) {
                if (left->grad.empty()) {
                    left->grad.assign(left->data.size(), 0.0f);
                }
                for (size_t i = 0; i < left->grad.size(); ++i) {
                    left->grad[i] += out_node->grad[i] / right->data[i];
                }
            }

            if (right->requires_grad) {
                if (right->grad.empty()) {
                    right->grad.assign(right->data.size(), 0.0f);
                }
                for (size_t i = 0; i < right->grad.size(); ++i) {
                    right->grad[i] -= out_node->grad[i] * left->data[i]
                                      / (right->data[i] * right->data[i]);
                }
            }
        };
    }

    return out;
}

Tensor Tensor::operator+(float v) const {
    auto out = map([v](float x){ return x + v; });

    if (out.node_->requires_grad) {
        // Keep shared_ptr only outside the lambda
        auto out_node_sp = out.node_;
        TensorNode* out_node = out_node_sp.get();   // raw pointer (no ownership)

        auto left = node_;

        out_node_sp->parents = { left };

        out_node_sp->backward = [out_node, left]() {
            if (!left->requires_grad) {
                return;
            }
            if (left->grad.empty()) {
                left->grad.assign(left->data.size(), 0.0f);
            }
            for (size_t i = 0; i < left->grad.size(); ++i) {
                left->grad[i] += out_node->grad[i];
            }
        };
    }

    return out;
}

Tensor Tensor::operator-(float v) const {
    auto out = map([v](float x){ return x - v; });

    if (out.node_->requires_grad) {
        // Keep shared_ptr only outside the lambda
        auto out_node_sp = out.node_;
        TensorNode* out_node = out_node_sp.get();   // raw pointer (no ownership)

        auto left = node_;

        out_node_sp->parents = { left };

        out_node_sp->backward = [out_node, left]() {
            if (!left->requires_grad) {
                return;
            }
            if (left->grad.empty()) {
                left->grad.assign(left->data.size(), 0.0f);
            }
            for (size_t i = 0; i < left->grad.size(); ++i) {
                left->grad[i] += out_node->grad[i];
            }
        };
    }

    return out;
}

Tensor Tensor::operator*(float v) const {
    auto out = map([v](float x){ return x * v; });

    if (out.node_->requires_grad) {
        // Keep shared_ptr only outside the lambda
        auto out_node_sp = out.node_;
        TensorNode* out_node = out_node_sp.get();   // raw pointer (no ownership)

        auto left = node_;

        out_node_sp->parents = { left };

        out_node_sp->backward = [out_node, left, v]() {
            if (!left->requires_grad) {
                return;
            }
            if (left->grad.empty()) {
                left->grad.assign(left->data.size(), 0.0f);
            }
            for (size_t i = 0; i < left->grad.size(); ++i) {
                left->grad[i] += out_node->grad[i] * v;
            }
        };
    }

    return out;
}

Tensor Tensor::operator/(float v) const {
    auto out = map([v](float x){ return x / v; });

    if (out.node_->requires_grad) {
        // Keep shared_ptr only outside the lambda
        auto out_node_sp = out.node_;
        TensorNode* out_node = out_node_sp.get();   // raw pointer (no ownership)

        auto left = node_;

        out_node_sp->parents = { left };

        out_node_sp->backward = [out_node, left, v]() {
            if (!left->requires_grad) {
                return;
            }
            if (left->grad.empty()) {
                left->grad.assign(left->data.size(), 0.0f);
            }
            for (size_t i = 0; i < left->grad.size(); ++i) {
                left->grad[i] += out_node->grad[i] / v;
            }
        };
    }

    return out;
}

Tensor Tensor::pow(float exponent) const {
    auto out = map([exponent](float x){ return std::pow(x, exponent); });

    if (out.node_->requires_grad) {
        // Hold shared_ptr only outside the lambda
        auto out_node_sp = out.node_;
        TensorNode* out_node = out_node_sp.get();   // raw pointer (no ownership)

        auto left = node_;

        out_node_sp->parents = { left };

        out_node_sp->backward = [out_node, left, exponent]() {
            if (!left->requires_grad) {
                return;
            }
            if (left->grad.empty()) {
                left->grad.assign(left->data.size(), 0.0f);
            }
            for (size_t i = 0; i < left->grad.size(); ++i) {
                left->grad[i] += out_node->grad[i]
                               * exponent
                               * std::pow(left->data[i], exponent - 1);
            }
        };
    }

    return out;
}

Tensor Tensor::softmax() const {
    // Softmax along the last dimension
    if (node_->shape.empty()) {
        throw std::invalid_argument("Softmax requires at least 1D tensor.");
    }
    
    size_t dim = node_->shape.back();
    size_t outer_size = node_->data.size() / dim;
    std::vector<float> result(node_->data.size());
 
    for (size_t i = 0; i < outer_size; ++i) {
        // Find max for numerical stability
        float max_val = node_->data[i * dim];
        for (size_t j = 1; j < dim; ++j) {
            if (node_->data[i * dim + j] > max_val) {
                max_val = node_->data[i * dim + j];
            }
        }
 
        // Compute exponentials and sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            result[i * dim + j] = std::exp(node_->data[i * dim + j] - max_val);
            sum_exp += result[i * dim + j];
        }
 
        // Normalize
        for (size_t j = 0; j < dim; ++j) {
            result[i * dim + j] /= sum_exp;
        }
    }
 
    Tensor out(std::move(result), node_->shape, node_->requires_grad);
 
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto left = node_;
        out_node->parents = {left};
        out_node->backward = [out_node, left, dim, outer_size]() {
            if (!left->requires_grad) {
                return;
            }
            if (left->grad.empty()) {
                left->grad.assign(left->data.size(), 0.0f);
            }
            // Softmax backward: d_input = softmax * (d_output - sum(d_output * softmax))
            for (size_t i = 0; i < outer_size; ++i) {
                // Compute dot product of gradient and softmax output for this row
                float dot = 0.0f;
                for (size_t j = 0; j < dim; ++j) {
                    dot += out_node->grad[i * dim + j] * out_node->data[i * dim + j];
                }
                // Compute gradient for each element
                for (size_t j = 0; j < dim; ++j) {
                    left->grad[i * dim + j] += out_node->data[i * dim + j] *
                        (out_node->grad[i * dim + j] - dot);
                }
            }
        };
    }
 
    return out;
}

Tensor Tensor::log_softmax() const {
    // Log-Softmax along the last dimension
    // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    if (node_->shape.empty()) {
        throw std::invalid_argument("Log-Softmax requires at least 1D tensor.");
    }
 
    size_t dim = node_->shape.back();
    size_t outer_size = node_->data.size() / dim;
    std::vector<float> result(node_->data.size());
 
    for (size_t i = 0; i < outer_size; ++i) {
        // Find max for numerical stability
        float max_val = node_->data[i * dim];
        for (size_t j = 1; j < dim; ++j) {
            if (node_->data[i * dim + j] > max_val) {
                max_val = node_->data[i * dim + j];
            }
        }
 
        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            sum_exp += std::exp(node_->data[i * dim + j] - max_val);
        }
 
        // Compute log_softmax: x - max - log(sum_exp)
        float log_sum_exp = std::log(sum_exp);
        for (size_t j = 0; j < dim; ++j) {
            result[i * dim + j] = node_->data[i * dim + j] - max_val - log_sum_exp;
        }
    }
 
    Tensor out(std::move(result), node_->shape, node_->requires_grad);
 
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto parent = node_;
        out_node->parents = {parent};
        out_node->backward = [out_node, parent, dim, outer_size]() {
            if (!parent->requires_grad) {
                return;
            }
            if (parent->grad.empty()) {
                parent->grad.assign(parent->data.size(), 0.0f);
            }
            // Gradient of log_softmax: grad_input = grad_output - softmax(x) * sum(grad_output)
            for (size_t i = 0; i < outer_size; ++i) {
                // Compute softmax for this row (from stored log_softmax output)
                // softmax = exp(log_softmax)
                float grad_sum = 0.0f;
                for (size_t j = 0; j < dim; ++j) {
                    grad_sum += out_node->grad[i * dim + j];
                }
                for (size_t j = 0; j < dim; ++j) {
                    float softmax_val = std::exp(out_node->data[i * dim + j]);
                    parent->grad[i * dim + j] += out_node->grad[i * dim + j] - softmax_val * grad_sum;
                }
            }
        };
    }
 
    return out;
}

// =============================================================
// Matrix Operations
// =============================================================

Tensor Tensor::matmul(const Tensor& other) const {
    if (node_->shape.size() != 2 || other.node_->shape.size() != 2) {
        throw std::invalid_argument("Matmul requires 2D tensors.");
    }
    if (node_->shape[1] != other.node_->shape[0]) {
        throw std::invalid_argument("Matmul dimension mismatch: (M,K) @ (K,N) required.");
    }

    size_t M = node_->shape[0];
    size_t K = node_->shape[1];
    size_t N = other.node_->shape[1];

    std::vector<float> result(M * N, 0.0f);

    // Simple O(N^3) implementation
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float val_a = node_->data[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                result[i * N + j] += val_a * other.node_->data[k * N + j];
            }
        }
    }
    Tensor out(std::move(result), {M, N}, node_->requires_grad || other.node_->requires_grad);
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto left = node_;
        auto right = other.node_;
        out_node->parents = {left, right};
        out_node->backward = [out_node, left, right, M, K, N]() {
            if (left->requires_grad) {
                if (left->grad.empty()) {
                    left->grad.assign(left->data.size(), 0.0f);
                }
                for (size_t i = 0; i < M; ++i) {
                    for (size_t k = 0; k < K; ++k) {
                        float accum = 0.0f;
                        for (size_t j = 0; j < N; ++j) {
                            accum += out_node->grad[i * N + j] * right->data[k * N + j];
                        }
                        left->grad[i * K + k] += accum;
                    }
                }
            }
            if (right->requires_grad) {
                if (right->grad.empty()) {
                    right->grad.assign(right->data.size(), 0.0f);
                }
                for (size_t k = 0; k < K; ++k) {
                    for (size_t j = 0; j < N; ++j) {
                        float accum = 0.0f;
                        for (size_t i = 0; i < M; ++i) {
                            accum += left->data[i * K + k] * out_node->grad[i * N + j];
                        }
                        right->grad[k * N + j] += accum;
                    }
                }
            }
        };
    }
    return out;
}

Tensor Tensor::transpose() const {
    if (node_->shape.size() != 2) throw std::invalid_argument("Transpose requires 2D tensor.");
    
    size_t R = node_->shape[0];
    size_t C = node_->shape[1];
    std::vector<float> result(node_->data.size());

    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            result[j * R + i] = node_->data[i * C + j];
        }
    }
    Tensor out(std::move(result), {C, R}, node_->requires_grad);
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto parent = node_;
        out_node->parents = {parent};
        out_node->backward = [out_node, parent, R, C]() {
            if (!parent->requires_grad) {
                return;
            }
            if (parent->grad.empty()) {
                parent->grad.assign(parent->data.size(), 0.0f);
            }
            for (size_t i = 0; i < R; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    parent->grad[i * C + j] += out_node->grad[j * R + i];
                }
            }
        };
    }
    return out;
}

// =============================================================
// Activations & Reductions
// =============================================================

Tensor Tensor::relu() const {
    auto out = map([](float x) { return std::max(0.0f, x); });
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto parent = node_;
        out_node->parents = {parent};
        out_node->backward = [out_node, parent]() {
            if (!parent->requires_grad) {
                return;
            }
            if (parent->grad.empty()) {
                parent->grad.assign(parent->data.size(), 0.0f);
            }
            for (size_t i = 0; i < parent->data.size(); ++i) {
                float grad = parent->data[i] > 0.0f ? out_node->grad[i] : 0.0f;
                parent->grad[i] += grad;
            }
        };
    }
    return out;
}

Tensor Tensor::sigmoid() const {
    auto out = map([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto parent = node_;
        out_node->parents = {parent};
        out_node->backward = [out_node, parent]() {
            if (!parent->requires_grad) {
                return;
            }
            if (parent->grad.empty()) {
                parent->grad.assign(parent->data.size(), 0.0f);
            }
            for (size_t i = 0; i < parent->data.size(); ++i) {
                float sigmoid_val = out_node->data[i];
                parent->grad[i] += out_node->grad[i] * sigmoid_val * (1.0f - sigmoid_val);
            }
        };
    }
    return out;
}

Tensor Tensor::tanh() const {
    auto out = map([](float x) { return std::tanh(x); });
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto parent = node_;
        out_node->parents = {parent};
        out_node->backward = [out_node, parent]() {
            if (!parent->requires_grad) {
                return;
            }
            if (parent->grad.empty()) {
                parent->grad.assign(parent->data.size(), 0.0f);
            }
            for (size_t i = 0; i < parent->data.size(); ++i) {
                float tanh_val = out_node->data[i];
                // d/dx tanh(x) = 1 - tanh^2(x)
                parent->grad[i] += out_node->grad[i] * (1.0f - tanh_val * tanh_val);
            }
        };
    }
    return out;
}
 
Tensor Tensor::log() const {
    auto out = map([](float x) { return std::log(x); });
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto parent = node_;
        out_node->parents = {parent};
        out_node->backward = [out_node, parent]() {
            if (!parent->requires_grad) {
                return;
            }
            if (parent->grad.empty()) {
                parent->grad.assign(parent->data.size(), 0.0f);
            }
            for (size_t i = 0; i < parent->data.size(); ++i) {
                // d/dx log(x) = 1/x
                parent->grad[i] += out_node->grad[i] / parent->data[i];
            }
        };
    }
    return out;
}
 
Tensor Tensor::clamp(float min_val, float max_val) const {
    auto out = map([min_val, max_val](float x) {
        return std::min(std::max(x, min_val), max_val);
    });
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto parent = node_;
        out_node->parents = {parent};
        out_node->backward = [out_node, parent, min_val, max_val]() {
            if (!parent->requires_grad) {
                return;
            }
            if (parent->grad.empty()) {
                parent->grad.assign(parent->data.size(), 0.0f);
            }
            for (size_t i = 0; i < parent->data.size(); ++i) {
                // Gradient is 1 if within bounds, 0 otherwise
                float x = parent->data[i];
                float grad = (x > min_val && x < max_val) ? out_node->grad[i] : 0.0f;
                parent->grad[i] += grad;
            }
        };
    }
    return out;
}
 
Tensor Tensor::leaky_relu(float negative_slope) const {
    auto out = map([negative_slope](float x) {
        return x > 0.0f ? x : negative_slope * x;
    });
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto parent = node_;
        out_node->parents = {parent};
        out_node->backward = [out_node, parent, negative_slope]() {
            if (!parent->requires_grad) {
                return;
            }
            if (parent->grad.empty()) {
                parent->grad.assign(parent->data.size(), 0.0f);
            }
            for (size_t i = 0; i < parent->data.size(); ++i) {
                float grad = parent->data[i] > 0.0f ? out_node->grad[i] : negative_slope * out_node->grad[i];
                parent->grad[i] += grad;
            }
        };
    }
    return out;
}

Tensor Tensor::sum() const {
    float total = std::accumulate(node_->data.begin(), node_->data.end(), 0.0f);
    Tensor out({total}, {1}, node_->requires_grad);
    if (out.node_->requires_grad) {
        auto out_node = out.node_;
        auto parent = node_;
        out_node->parents = {parent};
        out_node->backward = [out_node, parent]() {
            if (!parent->requires_grad) {
                return;
            }
            if (parent->grad.empty()) {
                parent->grad.assign(parent->data.size(), 0.0f);
            }
            for (size_t i = 0; i < parent->grad.size(); ++i) {
                parent->grad[i] += out_node->grad[0];
            }
        };
    }
    return out;
}

Tensor Tensor::mean() const {
    float scale = static_cast<float>(node_->data.size());
    Tensor out = sum() / scale;
    return out;
}

Tensor Tensor::std() const {
    float mean_val = mean().node_->data[0];
    float accum = 0.0f;
    for (const auto& x : node_->data) {
        accum += (x - mean_val) * (x - mean_val);
    }
    float variance = accum / static_cast<float>(node_->data.size());
    return Tensor({std::sqrt(variance)}, {1}, node_->requires_grad);
}

// =============================================================
// Autograd
// =============================================================

void Tensor::set_requires_grad(bool requires_grad) {
    node_->requires_grad = requires_grad;
    if (!requires_grad) {
        node_->grad.clear();
    }
}

bool Tensor::requires_grad() const {
    return node_->requires_grad;
}

const std::vector<float>& Tensor::grad() const {
    return node_->grad;
}

void Tensor::zero_grad() {
    ensure_grad(true);
}

void Tensor::backward() {
    if (!node_->requires_grad) {
        return;
    }
    if (node_->grad.empty()) {
        node_->grad.assign(node_->data.size(), 1.0f);
    }

    std::vector<std::shared_ptr<TensorNode>> topo;
    std::unordered_set<TensorNode*> visited;

    std::function<void(const std::shared_ptr<TensorNode>&)> build =
        [&](const std::shared_ptr<TensorNode>& n) {
            if (!n || visited.count(n.get()) > 0) {
                return;
            }
            visited.insert(n.get());
            for (const auto& parent : n->parents) {
                build(parent);
            }
            topo.push_back(n);
        };

    build(node_);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        auto& current = *it;
        if (current->backward) {
            current->backward();
        }
    }

    // After backward pass completes, release the graph
    for (auto& current : topo) {
        current->backward = nullptr;  // Release captured references
        current->parents.clear();     // Break parent reference chain
    }

}

// =============================================================
// Utilities
// =============================================================

const std::vector<float>& Tensor::data() const {
    return node_->data;
}

const std::vector<size_t>& Tensor::shape() const {
    return node_->shape;
}

float& Tensor::operator[](size_t i) {
    return node_->data[i];
}

const float& Tensor::operator[](size_t i) const {
    return node_->data[i];
}

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < node_->shape.size(); ++i) {
        std::cout << node_->shape[i];
        if (i < node_->shape.size() - 1) std::cout << ", ";
    }
    std::cout << "])\n";
    
    if (node_->shape.empty()) {
        std::cout << "[]\n";
        return;
    }
    
    print_recursive(0, 0, 0);
    std::cout << "\n";
}



void Tensor::print_recursive(size_t offset, size_t depth, size_t indent) const {
    // Base case: print 1D array (innermost dimension)
    if (depth == node_->shape.size() - 1) {
        std::cout << "[";
        size_t size = node_->shape[depth];
        size_t max_print = std::min(size, size_t(10));
        
        for (size_t i = 0; i < max_print; ++i) {
            std::cout << std::fixed << std::setprecision(4) << node_->data[offset + i];
            if (i < max_print - 1) std::cout << ", ";
        }
        
        if (size > max_print) {
            std::cout << ", ... (" << (size - max_print) << " more)";
        }
        std::cout << "]";
        return;
    }
    
    // Recursive case: print higher dimensions
    size_t current_dim_size = node_->shape[depth];
    size_t stride = 1;
    for (size_t i = depth + 1; i < node_->shape.size(); ++i) {
        stride *= node_->shape[i];
    }
    
    size_t max_print = std::min(current_dim_size, size_t(6));
    
    std::cout << "[";
    for (size_t i = 0; i < max_print; ++i) {
        if (i > 0) {
            std::cout << ",\n" << std::string(indent + 1, ' ');
        }
        
        print_recursive(offset + i * stride, depth + 1, indent + 1);
    }
    
    if (current_dim_size > max_print) {
        std::cout << ",\n" << std::string(indent + 1, ' ') << "...";
    }
    
    std::cout << "]";
}



} // namespace lamp