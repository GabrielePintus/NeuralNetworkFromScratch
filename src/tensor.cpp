/**
 * @file tensor.cpp
 * @brief Implementation of the Tensor class.
 */

#include "tensor.hpp"
#include <random>
#include <iomanip>

namespace lamp {

// =============================================================
// Constructor & Validation
// =============================================================

Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape)
    : data_(std::move(data)), shape_(std::move(shape)) {
    validate();
}

void Tensor::validate() const {
    size_t required_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
    if (data_.size() != required_size) {
        throw std::invalid_argument("Tensor Error: Data size (" + std::to_string(data_.size()) + 
                                  ") does not match shape capacity (" + std::to_string(required_size) + ")");
    }
}

// =============================================================
// Private Helpers
// =============================================================

Tensor Tensor::map(std::function<float(float)> op) const {
    std::vector<float> result = data_;
    std::transform(result.begin(), result.end(), result.begin(), op);
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::zip(const Tensor& other, std::function<float(float, float)> op) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor Error: Shape mismatch in element-wise operation.");
    }
    std::vector<float> result(data_.size());
    std::transform(data_.begin(), data_.end(), other.data_.begin(), result.begin(), op);
    return Tensor(std::move(result), shape_);
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
    std::fill(t.data_.begin(), t.data_.end(), 1.0f);
    return t;
}

Tensor Tensor::randn(const std::vector<size_t>& shape) {
    auto t = zeros(shape);
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : t.data_) x = dist(gen);
    return t;
}

Tensor Tensor::uniform(const std::vector<size_t>& shape, float low, float high) {
    auto t = zeros(shape);
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& x : t.data_) x = dist(gen);
    return t;
}

// =============================================================
// Operators
// =============================================================

Tensor Tensor::operator+(const Tensor& other) const {
    // If shapes match exactly, use element-wise addition
    if (shape_ == other.shape_) {
        return zip(other, std::plus<float>());
    }
    
    // Handle broadcasting: (M, N) + (N,)
    if (shape_.size() == 2 && other.shape_.size() == 1) {
        if (shape_[1] == other.shape_[0]) {
            return add_broadcast(other);
        }
    }
    
    // Handle broadcasting: (N,) + (M, N)
    if (shape_.size() == 1 && other.shape_.size() == 2) {
        if (shape_[0] == other.shape_[1]) {
            return other.add_broadcast(*this);
        }
    }
    
    throw std::invalid_argument(
        "Tensor Error: Incompatible shapes for addition. Broadcasting not supported for these shapes."
    );
}
Tensor Tensor::add_broadcast(const Tensor& vec) const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("add_broadcast: requires 2D tensor");
    }
    if (vec.shape_.size() != 1) {
        throw std::invalid_argument("add_broadcast: requires 1D vector");
    }
    if (shape_[1] != vec.shape_[0]) {
        throw std::invalid_argument("add_broadcast: vector size must match tensor columns");
    }
    
    size_t rows = shape_[0];
    size_t cols = shape_[1];
    std::vector<float> result = data_;
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i * cols + j] += vec.data_[j];
        }
    }
    
    return Tensor(std::move(result), shape_);
}
Tensor Tensor::operator-(const Tensor& o) const { return zip(o, std::minus<float>()); }
Tensor Tensor::operator*(const Tensor& o) const { return zip(o, std::multiplies<float>()); }
Tensor Tensor::operator/(const Tensor& o) const { return zip(o, std::divides<float>()); }

Tensor Tensor::operator+(float v) const { return map([v](float x){ return x + v; }); }
Tensor Tensor::operator-(float v) const { return map([v](float x){ return x - v; }); }
Tensor Tensor::operator*(float v) const { return map([v](float x){ return x * v; }); }
Tensor Tensor::operator/(float v) const { return map([v](float x){ return x / v; }); }
Tensor Tensor::softmax() const {
    // Softmax along the last dimension
    if (shape_.empty()) {
        throw std::invalid_argument("Softmax requires at least 1D tensor.");
    }
    
    size_t dim = shape_.back();
    size_t outer_size = data_.size() / dim;
    std::vector<float> result(data_.size());
    
    for (size_t i = 0; i < outer_size; ++i) {
        // Find max for numerical stability
        float max_val = data_[i * dim];
        for (size_t j = 1; j < dim; ++j) {
            if (data_[i * dim + j] > max_val) {
                max_val = data_[i * dim + j];
            }
        }
        
        // Compute exponentials and sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            result[i * dim + j] = std::exp(data_[i * dim + j] - max_val);
            sum_exp += result[i * dim + j];
        }
        
        // Normalize
        for (size_t j = 0; j < dim; ++j) {
            result[i * dim + j] /= sum_exp;
        }
    }
    
    return Tensor(std::move(result), shape_);
}

// =============================================================
// Matrix Operations
// =============================================================

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("Matmul requires 2D tensors.");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Matmul dimension mismatch: (M,K) @ (K,N) required.");
    }

    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = other.shape_[1];

    std::vector<float> result(M * N, 0.0f);

    // Simple O(N^3) implementation
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float val_a = data_[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                result[i * N + j] += val_a * other.data_[k * N + j];
            }
        }
    }
    return Tensor(std::move(result), {M, N});
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) throw std::invalid_argument("Transpose requires 2D tensor.");
    
    size_t R = shape_[0];
    size_t C = shape_[1];
    std::vector<float> result(data_.size());

    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            result[j * R + i] = data_[i * C + j];
        }
    }
    return Tensor(std::move(result), {C, R});
}

// =============================================================
// Activations & Reductions
// =============================================================

Tensor Tensor::relu() const {
    return map([](float x) { return std::max(0.0f, x); });
}

Tensor Tensor::sigmoid() const {
    return map([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}

Tensor Tensor::sum() const {
    float total = std::accumulate(data_.begin(), data_.end(), 0.0f);
    return Tensor({total}, {1});
}

Tensor Tensor::mean() const {
    return sum() / static_cast<float>(data_.size());
}

Tensor Tensor::std() const {
    float mean_val = mean().data_[0];
    float accum = 0.0f;
    for (const auto& x : data_) {
        accum += (x - mean_val) * (x - mean_val);
    }
    float variance = accum / static_cast<float>(data_.size());
    return Tensor({std::sqrt(variance)}, {1});
}

// =============================================================
// Utilities
// =============================================================

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "])\n";
    
    if (shape_.empty()) {
        std::cout << "[]\n";
        return;
    }
    
    print_recursive(0, 0, 0);
    std::cout << "\n";
}



void Tensor::print_recursive(size_t offset, size_t depth, size_t indent) const {
    // Base case: print 1D array (innermost dimension)
    if (depth == shape_.size() - 1) {
        std::cout << "[";
        size_t size = shape_[depth];
        size_t max_print = std::min(size, size_t(10));
        
        for (size_t i = 0; i < max_print; ++i) {
            std::cout << std::fixed << std::setprecision(4) << data_[offset + i];
            if (i < max_print - 1) std::cout << ", ";
        }
        
        if (size > max_print) {
            std::cout << ", ... (" << (size - max_print) << " more)";
        }
        std::cout << "]";
        return;
    }
    
    // Recursive case: print higher dimensions
    size_t current_dim_size = shape_[depth];
    size_t stride = 1;
    for (size_t i = depth + 1; i < shape_.size(); ++i) {
        stride *= shape_[i];
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