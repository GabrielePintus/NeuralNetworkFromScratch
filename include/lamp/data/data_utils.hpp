#pragma once
 
#include "lamp/core/tensor.hpp"
#include <utility>
 
namespace lamp {
namespace data {
 
/**
 * @brief Generate a simple regression dataset.
 *
 * Creates X and y where y = X @ weights + bias + noise
 *
 * @param n_samples Number of samples to generate.
 * @param n_features Number of input features.
 * @param noise Standard deviation of Gaussian noise added to targets.
 * @return std::pair<Tensor, Tensor> (X, y) pair where X is (n_samples, n_features)
 *         and y is (n_samples, 1).
 */
std::pair<Tensor, Tensor> make_regression(
    size_t n_samples,
    size_t n_features = 1,
    float noise = 0.1f
);
 
/**
 * @brief Generate a simple binary classification dataset.
 *
 * Creates two clusters of points, one for each class.
 *
 * @param n_samples Number of samples to generate.
 * @param n_features Number of input features.
 * @param separation Distance between cluster centers.
 * @return std::pair<Tensor, Tensor> (X, y) pair where X is (n_samples, n_features)
 *         and y is (n_samples, 1) with values 0 or 1.
 */
std::pair<Tensor, Tensor> make_classification(
    size_t n_samples,
    size_t n_features = 2,
    float separation = 2.0f
);
 
/**
 * @brief Generate a multi-class classification dataset.
 *
 * Creates n_classes clusters of points.
 *
 * @param n_samples Number of samples to generate.
 * @param n_features Number of input features.
 * @param n_classes Number of classes.
 * @param separation Distance between cluster centers.
 * @return std::pair<Tensor, Tensor> (X, y) pair where X is (n_samples, n_features)
 *         and y is (n_samples, n_classes) one-hot encoded.
 */
std::pair<Tensor, Tensor> make_multiclass(
    size_t n_samples,
    size_t n_features = 2,
    size_t n_classes = 3,
    float separation = 2.0f
);
 
/**
 * @brief Generate a "moons" shaped dataset for binary classification.
 *
 * Creates two interleaving half-circle shapes.
 *
 * @param n_samples Number of samples to generate.
 * @param noise Standard deviation of Gaussian noise.
 * @return std::pair<Tensor, Tensor> (X, y) pair.
 */
std::pair<Tensor, Tensor> make_moons(
    size_t n_samples,
    float noise = 0.1f
);
 
/**
 * @brief Generate a spiral dataset for classification.
 *
 * Creates n_classes interleaving spirals.
 *
 * @param n_samples Number of samples to generate.
 * @param n_classes Number of spiral arms (classes).
 * @param noise Standard deviation of Gaussian noise.
 * @return std::pair<Tensor, Tensor> (X, y) pair where y is one-hot encoded.
 */
std::pair<Tensor, Tensor> make_spirals(
    size_t n_samples,
    size_t n_classes = 2,
    float noise = 0.1f
);
 
} // namespace data
} // namespace lamp