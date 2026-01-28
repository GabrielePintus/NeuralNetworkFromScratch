#include "data_utils.hpp"
#include <random>
#include <cmath>
 
namespace lamp {
namespace data {
 
// Static random engine for reproducibility within session
static std::mt19937& get_rng() {
    static std::mt19937 rng(std::random_device{}());
    return rng;
}
 
std::pair<Tensor, Tensor> make_regression(
    size_t n_samples,
    size_t n_features,
    float noise
) {
    auto& rng = get_rng();
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
 
    // Generate random weights and bias for ground truth
    std::vector<float> true_weights(n_features);
    for (size_t i = 0; i < n_features; ++i) {
        true_weights[i] = uniform(rng) * 3.0f;  // Scale for interesting values
    }
    float true_bias = uniform(rng) * 2.0f;
 
    // Generate X
    std::vector<float> x_data(n_samples * n_features);
    for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = normal(rng);
    }
 
    // Generate y = X @ weights + bias + noise
    std::vector<float> y_data(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        float y_val = true_bias;
        for (size_t j = 0; j < n_features; ++j) {
            y_val += x_data[i * n_features + j] * true_weights[j];
        }
        y_val += normal(rng) * noise;
        y_data[i] = y_val;
    }
 
    Tensor X(x_data, {n_samples, n_features}, false);
    Tensor y(y_data, {n_samples, 1}, false);
 
    return {X, y};
}
 
std::pair<Tensor, Tensor> make_classification(
    size_t n_samples,
    size_t n_features,
    float separation
) {
    auto& rng = get_rng();
    std::normal_distribution<float> normal(0.0f, 1.0f);
 
    size_t n_per_class = n_samples / 2;
    size_t remainder = n_samples % 2;
 
    std::vector<float> x_data(n_samples * n_features);
    std::vector<float> y_data(n_samples);
 
    // Class 0: centered at -separation/2
    for (size_t i = 0; i < n_per_class; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            x_data[i * n_features + j] = normal(rng) - separation / 2.0f;
        }
        y_data[i] = 0.0f;
    }
 
    // Class 1: centered at +separation/2
    for (size_t i = n_per_class; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            x_data[i * n_features + j] = normal(rng) + separation / 2.0f;
        }
        y_data[i] = 1.0f;
    }
 
    Tensor X(x_data, {n_samples, n_features}, false);
    Tensor y(y_data, {n_samples, 1}, false);
 
    return {X, y};
}
 
std::pair<Tensor, Tensor> make_multiclass(
    size_t n_samples,
    size_t n_features,
    size_t n_classes,
    float separation
) {
    auto& rng = get_rng();
    std::normal_distribution<float> normal(0.0f, 0.5f);
 
    size_t n_per_class = n_samples / n_classes;
 
    std::vector<float> x_data(n_samples * n_features);
    std::vector<float> y_data(n_samples * n_classes, 0.0f);  // One-hot encoded
 
    // Generate cluster centers in a circle
    std::vector<std::vector<float>> centers(n_classes, std::vector<float>(n_features, 0.0f));
    for (size_t c = 0; c < n_classes; ++c) {
        float angle = 2.0f * 3.14159265f * static_cast<float>(c) / static_cast<float>(n_classes);
        centers[c][0] = separation * std::cos(angle);
        if (n_features > 1) {
            centers[c][1] = separation * std::sin(angle);
        }
    }
 
    size_t sample_idx = 0;
    for (size_t c = 0; c < n_classes; ++c) {
        size_t samples_for_class = (c == n_classes - 1) ? (n_samples - sample_idx) : n_per_class;
 
        for (size_t i = 0; i < samples_for_class && sample_idx < n_samples; ++i, ++sample_idx) {
            for (size_t j = 0; j < n_features; ++j) {
                x_data[sample_idx * n_features + j] = centers[c][j] + normal(rng);
            }
            // One-hot encoding
            y_data[sample_idx * n_classes + c] = 1.0f;
        }
    }
 
    Tensor X(x_data, {n_samples, n_features}, false);
    Tensor y(y_data, {n_samples, n_classes}, false);
 
    return {X, y};
}
 
std::pair<Tensor, Tensor> make_moons(
    size_t n_samples,
    float noise
) {
    auto& rng = get_rng();
    std::normal_distribution<float> normal(0.0f, noise);
 
    size_t n_per_moon = n_samples / 2;
    const float pi = 3.14159265f;
 
    std::vector<float> x_data(n_samples * 2);  // 2D points
    std::vector<float> y_data(n_samples);
 
    // First moon (top)
    for (size_t i = 0; i < n_per_moon; ++i) {
        float theta = pi * static_cast<float>(i) / static_cast<float>(n_per_moon);
        x_data[i * 2 + 0] = std::cos(theta) + normal(rng);
        x_data[i * 2 + 1] = std::sin(theta) + normal(rng);
        y_data[i] = 0.0f;
    }
 
    // Second moon (bottom, shifted)
    for (size_t i = n_per_moon; i < n_samples; ++i) {
        float theta = pi * static_cast<float>(i - n_per_moon) / static_cast<float>(n_samples - n_per_moon);
        x_data[i * 2 + 0] = 1.0f - std::cos(theta) + normal(rng);
        x_data[i * 2 + 1] = 0.5f - std::sin(theta) + normal(rng);
        y_data[i] = 1.0f;
    }
 
    Tensor X(x_data, {n_samples, 2}, false);
    Tensor y(y_data, {n_samples, 1}, false);
 
    return {X, y};
}
 
std::pair<Tensor, Tensor> make_spirals(
    size_t n_samples,
    size_t n_classes,
    float noise
) {
    auto& rng = get_rng();
    std::normal_distribution<float> normal(0.0f, noise);
 
    // Ensure at least 2 classes
    if (n_classes < 2) n_classes = 2;
 
    size_t n_per_class = n_samples / n_classes;
    const float pi = 3.14159265f;
 
    std::vector<float> x_data(n_samples * 2);  // 2D points
 
    // For binary classification, use simple labels; otherwise one-hot
    bool binary = (n_classes == 2);
    std::vector<float> y_data;
    if (binary) {
        y_data.resize(n_samples, 0.0f);
    } else {
        y_data.resize(n_samples * n_classes, 0.0f);
    }
 
    size_t sample_idx = 0;
    for (size_t c = 0; c < n_classes; ++c) {
        size_t samples_for_class = (c == n_classes - 1) ? (n_samples - sample_idx) : n_per_class;
        float delta_theta = 2.0f * pi / static_cast<float>(n_classes);
 
        for (size_t i = 0; i < samples_for_class && sample_idx < n_samples; ++i, ++sample_idx) {
            float t = static_cast<float>(i) / static_cast<float>(n_per_class);
            float r = t * 5.0f;  // Radius grows with t
            float theta = t * 4.0f * pi + delta_theta * static_cast<float>(c);
 
            x_data[sample_idx * 2 + 0] = r * std::cos(theta) + normal(rng);
            x_data[sample_idx * 2 + 1] = r * std::sin(theta) + normal(rng);
 
            if (binary) {
                y_data[sample_idx] = static_cast<float>(c);
            } else {
                y_data[sample_idx * n_classes + c] = 1.0f;
            }
        }
    }
 
    Tensor X(x_data, {n_samples, 2}, false);
    Tensor y = binary ? Tensor(y_data, {n_samples, 1}, false)
                      : Tensor(y_data, {n_samples, n_classes}, false);
 
    return {X, y};
}
 
} // namespace data
} // namespace lamp
 