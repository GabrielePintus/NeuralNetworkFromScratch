/**
 * @file tensor.hpp
 * @brief A simple Tensor library for C++ with autograd support.
 * @author Lamp Project
 * @version 1.0
 */

#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>

namespace lamp {

/**
 * @class Tensor
 * @brief A multi-dimensional array class for numerical operations with autograd support.
 * * This class provides a basic implementation of a tensor supporting:
 * - Element-wise arithmetic
 * - Matrix multiplication
 * - Activation functions (ReLU, Sigmoid)
 * - Basic statistical reductions
 */
class Tensor {
public:
    /**
     * @brief Construct a new Tensor object.
     * * @param data A flat vector containing the numeric values.
     * @param shape A vector describing the dimensions (e.g., {rows, cols}).
     * @throw std::invalid_argument If data size does not match the product of shape dimensions.
     */
    Tensor(std::vector<float> data, std::vector<size_t> shape, bool requires_grad = false);

    // =============================================================
    // Factory Methods
    // =============================================================

    /**
     * @brief Creates a tensor filled with zeros.
     * @param shape The target dimensions.
     * @return Tensor A tensor initialized with 0.0f.
     */
    static Tensor zeros(const std::vector<size_t>& shape);

    /**
     * @brief Creates a tensor filled with ones.
     * @param shape The target dimensions.
     * @return Tensor A tensor initialized with 1.0f.
     */
    static Tensor ones(const std::vector<size_t>& shape);

    /**
     * @brief Creates a tensor with random values from a Normal Distribution.
     * @param shape The target dimensions.
     * @return Tensor A tensor with values ~ N(0, 1).
     */
    static Tensor randn(const std::vector<size_t>& shape);

    /**
     * @brief Creates a tensor with random values from a Uniform Distribution.
     * @param shape The target dimensions.
     * @param low The lower bound (inclusive).
     * @param high The upper bound (exclusive).
     * @return Tensor A tensor with values in [low, high).
     */
    static Tensor uniform(const std::vector<size_t>& shape, float low = 0.0f, float high = 1.0f);

    // =============================================================
    // Arithmetic Operators
    // =============================================================

    /**
     * @brief Element-wise addition.
     * @param other The tensor to add.
     * @return Tensor Result of (this + other).
     */
    Tensor operator+(const Tensor& other) const;

    /**
     * @brief Add a 1D tensor (vector) to each row of a 2D tensor (broadcasting).
     * @param vec The 1D tensor to broadcast.
     * @return Tensor Result with vec added to each row.
     */
    Tensor add_broadcast(const Tensor& vec) const;

    /**
     * @brief Element-wise subtraction.
     * @param other The tensor to subtract.
     * @return Tensor Result of (this - other).
     */
    Tensor operator-(const Tensor& other) const;

    /**
     * @brief Element-wise multiplication (Hadamard product).
     * @param other The tensor to multiply.
     * @return Tensor Result of (this * other).
     */
    Tensor operator*(const Tensor& other) const;

    /**
     * @brief Element-wise division.
     * @param other The denominator tensor.
     * @return Tensor Result of (this / other).
     */
    Tensor operator/(const Tensor& other) const;

    // =============================================================
    // Scalar Operators
    // =============================================================

    /**
     * @brief Add a scalar to every element.
     * @param val The scalar value.
     */
    Tensor operator+(float val) const;

    /**
     * @brief Subtract a scalar from every element.
     * @param val The scalar value.
     */
    Tensor operator-(float val) const;

    /**
     * @brief Multiply every element by a scalar.
     * @param val The scalar value.
     */
    Tensor operator*(float val) const;

    /**
     * @brief Divide every element by a scalar.
     * @param val The scalar value.
     */
    Tensor operator/(float val) const;

    /**
     * @brief Element-wise exponentiation.
     * @param power The exponent value.
     * @return Tensor Result of raising each element to the given power.
     */
    Tensor pow(float power) const;

    /**
     * @brief Applies the Softmax function along the last dimension.
     * @return Tensor The softmax-normalized tensor.
     */
    Tensor softmax() const;

    // =============================================================
    // Matrix Operations
    // =============================================================

    /**
     * @brief Performs Matrix Multiplication.
     * * Requires both tensors to be 2-dimensional.
     * Shapes must satisfy (M, K) @ (K, N) -> (M, N).
     * * @param other The right-hand side operand.
     * @return Tensor The resulting matrix product.
     * @throw std::invalid_argument If shapes are incompatible or tensors are not 2D.
     */
    Tensor matmul(const Tensor& other) const;

    /**
     * @brief Transposes a 2D tensor.
     * Switches dimensions from (rows, cols) to (cols, rows).
     * @return Tensor The transposed tensor.
     */
    Tensor transpose() const;

    // =============================================================
    // Activations & Reductions
    // =============================================================

    /**
     * @brief Applies Rectified Linear Unit function element-wise.
     * f(x) = max(0, x)
     * @return Tensor
     */
    Tensor relu() const;

    /**
     * @brief Applies Sigmoid function element-wise.
     * f(x) = 1 / (1 + exp(-x))
     * @return Tensor
     */
    Tensor sigmoid() const;

    /**
     * @brief Sums all elements in the tensor.
     * @return Tensor A 1D tensor of size 1 containing the sum.
     */
    Tensor sum() const;

    /**
     * @brief Computes the mean of all elements.
     * @return Tensor A 1D tensor of size 1 containing the mean.
     */
    Tensor mean() const;

    /**
     * @brief Computes the standard deviation of all elements.
     * @return Tensor A 1D tensor of size 1 containing the standard deviation.
     */
    Tensor std() const;

    // =============================================================
    // Autograd
    // =============================================================

    /**
     * @brief Enable or disable gradient tracking.
     * @param requires_grad True to track gradients.
     */
    void set_requires_grad(bool requires_grad);

    /**
     * @brief Returns true if this tensor tracks gradients.
     */
    bool requires_grad() const;

    /**
     * @brief Access the gradient buffer.
     * @return const std::vector<float>& 
     */
    const std::vector<float>& grad() const;

    /**
     * @brief Zeros out the gradient buffer.
     */
    void zero_grad();

    /**
     * @brief Runs backpropagation starting from this tensor.
     * * If no gradient exists, seeds with ones.
     */
    void backward();

    // =============================================================
    // Utilities & Accessors
    // =============================================================

    /**
     * @brief Prints the tensor shape and data to stdout.
     */
    void print() const;

    /**
     * @brief Get the underlying flat data vector.
     * @return const std::vector<float>& 
     */
    const std::vector<float>& data() const;

    /**
     * @brief Get the shape vector.
     * @return const std::vector<size_t>& 
     */
    const std::vector<size_t>& shape() const;

    /**
     * @brief Access element by flat index (mutable).
     * @param i Index.
     * @return float& Reference to value.
     */
    float& operator[](size_t i);

    /**
     * @brief Access element by flat index (const).
     * @param i Index.
     * @return const float& Value.
     */
    const float& operator[](size_t i) const;

private:
    struct TensorNode {
        std::vector<float> data;
        std::vector<size_t> shape;
        std::vector<float> grad;
        bool requires_grad = false;
        std::function<void()> backward;
        std::vector<std::shared_ptr<TensorNode>> parents;
    };

    std::shared_ptr<TensorNode> node_; ///< Shared storage for tensor data + autograd

    /**
     * @brief Validates that data size matches shape dimensions.
     * @throw std::invalid_argument if mismatch.
     */
    void validate() const;

    /**
     * @brief Ensure the gradient buffer exists and is zeroed if requested.
     */
    void ensure_grad(bool zero = false);

    /**
     * @brief Applies a unary function to every element.
     * @param op Function taking float returning float.
     * @return Tensor New tensor with mapped values.
     */
    Tensor map(std::function<float(float)> op) const;

    /**
     * @brief Applies a binary function to two tensors element-wise.
     * @param other The second tensor.
     * @param op Function taking two floats returning float.
     * @return Tensor New tensor with combined values.
     */
    Tensor zip(const Tensor& other, std::function<float(float, float)> op) const;

    /**
     * @brief Recursively prints tensor in nested bracket notation.
     * @param offset Starting index in flat data array.
     * @param depth Current dimension depth.
     * @param indent Indentation level for formatting.
     */
    void print_recursive(size_t offset, size_t depth, size_t indent) const;
};

} // namespace lamp