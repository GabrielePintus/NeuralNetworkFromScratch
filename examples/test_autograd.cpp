#include "linear.hpp"
#include "optimizer.hpp"
#include "sequential.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include "data_utils.hpp"
#include "dataloader.hpp"
 
#include <iostream>
#include <memory>
#include <string>
#include <vector>
 
namespace {
 
void print_progress(size_t epoch, size_t total_epochs, float avg_loss, float accuracy) {
    const int bar_width = 40;
    float progress = float(epoch) / total_epochs;
    int pos = int(bar_width * progress);
 
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
 
    std::cout << "] " << int(progress * 100.0f)
              << "%  Loss: " << avg_loss
              << "  Acc: " << int(accuracy * 100.0f) << "%";
 
    std::cout.flush();
}
 
float compute_accuracy(const lamp::Tensor& preds, const lamp::Tensor& targets) {
    size_t correct = 0;
    size_t total = targets.shape()[0];
    size_t n_classes = targets.shape()[1];

    const float* preds_data = preds.data().data();
    const float* targets_data = targets.data().data();

    for (size_t i = 0; i < total; ++i) {
        size_t pred_class = 0;
        size_t true_class = 0;

        float max_pred = preds_data[i * n_classes];
        float max_target = targets_data[i * n_classes];

        for (size_t j = 1; j < n_classes; ++j) {
            float p = preds_data[i * n_classes + j];
            float t = targets_data[i * n_classes + j];

            if (p > max_pred) {
                max_pred = p;
                pred_class = j;
            }
            if (t > max_target) {
                max_target = t;
                true_class = j;
            }
        }

        if (pred_class == true_class)
            ++correct;
    }

    return float(correct) / float(total);
}
 
} // namespace
 
int main() {
    using namespace lamp;
    using namespace lamp::data;
 
    std::cout << "=== Moons Classification with DataLoader ===\n\n";
 
    // Generate moons dataset
    const size_t n_samples = 2000;
    const size_t n_classes = 5;
    const float noise = 0.1f;
    // auto [X, y] = make_moons(n_samples, noise);
    auto [X, y] = make_spirals(n_samples, n_classes, noise);
 
    std::cout << "Generated " << n_samples << " spiral samples\n";
    std::cout << "X shape: (" << X.shape()[0] << ", " << X.shape()[1] << ")\n";
    std::cout << "y shape: (" << y.shape()[0] << ", " << y.shape()[1] << ")\n\n";
 
    // Create dataset and dataloader
    const size_t batch_size = 16;
    auto dataset = std::make_shared<TensorDataset>(X, y);
    DataLoader dataloader(dataset, batch_size, true);  // shuffle=true
 
    std::cout << "DataLoader: " << dataloader.num_batches() << " batches of size " << batch_size << "\n\n";
 
    Sequential model({
        std::make_shared<Linear>(2, 64),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(64, 32),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(32, n_classes)
    });
 
    // Loss and optimizer
    CrossEntropyLoss criterion;
    Adam optimizer(model, 0.001f);
 
    // Training loop
    const size_t epochs = 500;
 
    std::cout << "Training for " << epochs << " epochs...\n";
 
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        size_t num_batches = 0;
 
        // Iterate over batches
        for (const auto& batch : dataloader) {
            // Forward pass
            Tensor preds = model(batch.X);
 
            // Compute loss
            Tensor loss = criterion(preds, batch.y);
 
            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
 
            epoch_loss += loss.data()[0];
            num_batches++;
        }
 
        // Compute average loss and full dataset accuracy
        float avg_loss = epoch_loss / static_cast<float>(num_batches);
        Tensor full_preds = model.forward(X);
        float accuracy = compute_accuracy(full_preds, y);
 
        print_progress(epoch + 1, epochs, avg_loss, accuracy);
 
        // Reset dataloader for next epoch (reshuffles data)
        dataloader.reset();
    }
 
    std::cout << "\n\nTraining complete!\n";
 
    // Final evaluation
    Tensor final_preds = model.forward(X);
    float final_accuracy = compute_accuracy(final_preds, y);
    Tensor final_loss = criterion.forward(final_preds, y);
 
    std::cout << "\nFinal Results:\n";
    std::cout << "  Loss:     " << final_loss.data()[0] << "\n";
    std::cout << "  Accuracy: " << int(final_accuracy * 100.0f) << "%\n";
 
    // Show some predictions
    std::cout << "\nSample predictions (first 10):\n";
    std::cout << "  X1\t\tX2\t\tTrue\tPred\n";
    for (size_t i = 0; i < 10; ++i) {
        float x1 = X.data()[i * 2];
        float x2 = X.data()[i * 2 + 1];
        // Find true class (argmax of one-hot target)
        size_t true_class = 0;
        for (size_t j = 1; j < n_classes; ++j) {
            if (y.data()[i * n_classes + j] > y.data()[i * n_classes + true_class]) {
                true_class = j;
            }
        }
 
        // Find predicted class (argmax of logits)
        size_t pred_class = 0;
        for (size_t j = 1; j < n_classes; ++j) {
            if (final_preds.data()[i * n_classes + j] > final_preds.data()[i * n_classes + pred_class]) {
                pred_class = j;
            }
        }
 
        std::cout << "  " << x1 << "\t" << x2 << "\t"
                  << true_class << "\t" << pred_class << "\n";
    }
 
    return 0;
}
 