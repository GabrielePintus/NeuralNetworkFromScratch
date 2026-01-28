#pragma once
 
#include "lamp/core/tensor.hpp"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
 
namespace lamp {
namespace data {
 
/**
 * @brief Abstract base class for datasets.
 *
 * A Dataset wraps data and provides access to individual samples.
 */
class Dataset {
public:
    virtual ~Dataset() = default;
 
    /**
     * @brief Get the number of samples in the dataset.
     */
    virtual size_t size() const = 0;
 
    /**
     * @brief Get a single sample by index.
     * @param index The sample index.
     * @return std::pair<Tensor, Tensor> (input, target) pair.
     */
    virtual std::pair<Tensor, Tensor> get(size_t index) const = 0;
};
 
/**
 * @brief A dataset that wraps Tensor data.
 *
 * Stores X and y tensors and provides sample-level access.
 */
class TensorDataset : public Dataset {
public:
    /**
     * @brief Construct a TensorDataset from X and y tensors.
     * @param X Input tensor of shape (n_samples, ...).
     * @param y Target tensor of shape (n_samples, ...).
     */
    TensorDataset(const Tensor& X, const Tensor& y);
 
    size_t size() const override;
    std::pair<Tensor, Tensor> get(size_t index) const override;
 
    /**
     * @brief Get the full X tensor.
     */
    const Tensor& X() const { return X_; }
 
    /**
     * @brief Get the full y tensor.
     */
    const Tensor& y() const { return y_; }
 
private:
    Tensor X_;
    Tensor y_;
    size_t n_samples_;
    size_t x_sample_size_;
    size_t y_sample_size_;
};
 
/**
 * @brief A batch of data returned by the DataLoader.
 */
struct Batch {
    Tensor X;
    Tensor y;
    size_t size;  // Actual batch size (may be smaller for last batch)
};
 
/**
 * @brief DataLoader for iterating over datasets in batches.
 *
 * Supports shuffling and configurable batch sizes.
 */
class DataLoader {
public:
    /**
     * @brief Construct a DataLoader.
     * @param dataset Shared pointer to the dataset.
     * @param batch_size Number of samples per batch.
     * @param shuffle Whether to shuffle indices each epoch.
     */
    DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle = true);
 
    /**
     * @brief Iterator class for range-based for loops.
     */
    class Iterator {
    public:
        Iterator(DataLoader* loader, size_t batch_idx);
 
        Batch operator*();
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
 
    private:
        DataLoader* loader_;
        size_t batch_idx_;
    };
 
    /**
     * @brief Get iterator to first batch.
     */
    Iterator begin();
 
    /**
     * @brief Get iterator past last batch.
     */
    Iterator end();
 
    /**
     * @brief Get the number of batches per epoch.
     */
    size_t num_batches() const;
 
    /**
     * @brief Reset and optionally shuffle the data for a new epoch.
     */
    void reset();
 
    /**
     * @brief Get a specific batch by index.
     */
    Batch get_batch(size_t batch_idx);
 
private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
    std::mt19937 rng_;
};
 
} // namespace data
} // namespace lamp