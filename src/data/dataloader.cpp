#include "lamp/data/dataloader.hpp"
#include <stdexcept>
#include <numeric>
 
namespace lamp {
namespace data {
 
// =============================================================
// TensorDataset
// =============================================================
 
TensorDataset::TensorDataset(const Tensor& X, const Tensor& y)
    : X_(X), y_(y) {
    if (X.shape().empty() || y.shape().empty()) {
        throw std::invalid_argument("TensorDataset: X and y must have at least one dimension");
    }
 
    n_samples_ = X.shape()[0];
    if (y.shape()[0] != n_samples_) {
        throw std::invalid_argument("TensorDataset: X and y must have same number of samples");
    }
 
    // Calculate sample sizes
    x_sample_size_ = 1;
    for (size_t i = 1; i < X.shape().size(); ++i) {
        x_sample_size_ *= X.shape()[i];
    }
 
    y_sample_size_ = 1;
    for (size_t i = 1; i < y.shape().size(); ++i) {
        y_sample_size_ *= y.shape()[i];
    }
}
 
size_t TensorDataset::size() const {
    return n_samples_;
}
 
std::pair<Tensor, Tensor> TensorDataset::get(size_t index) const {
    if (index >= n_samples_) {
        throw std::out_of_range("TensorDataset: index out of range");
    }
 
    // Extract single sample from X
    std::vector<float> x_sample(x_sample_size_);
    std::vector<size_t> x_shape;
    for (size_t i = 1; i < X_.shape().size(); ++i) {
        x_shape.push_back(X_.shape()[i]);
    }
    if (x_shape.empty()) {
        x_shape.push_back(1);
    }
 
    for (size_t i = 0; i < x_sample_size_; ++i) {
        x_sample[i] = X_.data()[index * x_sample_size_ + i];
    }
 
    // Extract single sample from y
    std::vector<float> y_sample(y_sample_size_);
    std::vector<size_t> y_shape;
    for (size_t i = 1; i < y_.shape().size(); ++i) {
        y_shape.push_back(y_.shape()[i]);
    }
    if (y_shape.empty()) {
        y_shape.push_back(1);
    }
 
    for (size_t i = 0; i < y_sample_size_; ++i) {
        y_sample[i] = y_.data()[index * y_sample_size_ + i];
    }
 
    return {Tensor(x_sample, x_shape, false), Tensor(y_sample, y_shape, false)};
}
 
// =============================================================
// DataLoader
// =============================================================
 
DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle)
    : dataset_(dataset),
      batch_size_(batch_size),
      shuffle_(shuffle),
      rng_(std::random_device{}()) {
 
    indices_.resize(dataset_->size());
    std::iota(indices_.begin(), indices_.end(), 0);
 
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}
 
size_t DataLoader::num_batches() const {
    return (dataset_->size() + batch_size_ - 1) / batch_size_;
}
 
void DataLoader::reset() {
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}
 
Batch DataLoader::get_batch(size_t batch_idx) {
    size_t start_idx = batch_idx * batch_size_;
    size_t end_idx = std::min(start_idx + batch_size_, dataset_->size());
    size_t actual_batch_size = end_idx - start_idx;
 
    if (start_idx >= dataset_->size()) {
        throw std::out_of_range("DataLoader: batch index out of range");
    }
 
    // Get first sample to determine shapes
    auto [first_x, first_y] = dataset_->get(indices_[start_idx]);
 
    // Calculate output shapes
    std::vector<size_t> x_shape = {actual_batch_size};
    for (size_t dim : first_x.shape()) {
        x_shape.push_back(dim);
    }
 
    std::vector<size_t> y_shape = {actual_batch_size};
    for (size_t dim : first_y.shape()) {
        y_shape.push_back(dim);
    }
 
    // Collect all samples
    size_t x_sample_size = first_x.data().size();
    size_t y_sample_size = first_y.data().size();
 
    std::vector<float> x_data(actual_batch_size * x_sample_size);
    std::vector<float> y_data(actual_batch_size * y_sample_size);
 
    // Copy first sample
    std::copy(first_x.data().begin(), first_x.data().end(), x_data.begin());
    std::copy(first_y.data().begin(), first_y.data().end(), y_data.begin());
 
    // Copy remaining samples
    for (size_t i = 1; i < actual_batch_size; ++i) {
        auto [x_i, y_i] = dataset_->get(indices_[start_idx + i]);
        std::copy(x_i.data().begin(), x_i.data().end(), x_data.begin() + i * x_sample_size);
        std::copy(y_i.data().begin(), y_i.data().end(), y_data.begin() + i * y_sample_size);
    }
 
    return Batch{
        Tensor(x_data, x_shape, false),
        Tensor(y_data, y_shape, false),
        actual_batch_size
    };
}
 
// =============================================================
// DataLoader::Iterator
// =============================================================
 
DataLoader::Iterator::Iterator(DataLoader* loader, size_t batch_idx)
    : loader_(loader), batch_idx_(batch_idx) {}
 
Batch DataLoader::Iterator::operator*() {
    return loader_->get_batch(batch_idx_);
}
 
DataLoader::Iterator& DataLoader::Iterator::operator++() {
    ++batch_idx_;
    return *this;
}
 
bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return batch_idx_ != other.batch_idx_;
}
 
DataLoader::Iterator DataLoader::begin() {
    return Iterator(this, 0);
}
 
DataLoader::Iterator DataLoader::end() {
    return Iterator(this, num_batches());
}
 
} // namespace data
} // namespace lamp
 