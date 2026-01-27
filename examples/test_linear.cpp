#include "linear.hpp"
#include <iostream>

using namespace lamp;

int main() {
    // Create a simple 2-layer MLP
    Linear fc1(784, 128);  // Input: 784 (28x28), Hidden: 128
    Linear fc2(128, 10);   // Hidden: 128, Output: 10 classes
    
    // Create random input (batch_size=32, features=784)
    Tensor x = Tensor::randn({32, 784});
    
    // Forward pass
    Tensor h = fc1(x).relu();      // Hidden layer with ReLU
    Tensor logits = fc2(h);        // Output layer
    Tensor probs = logits.softmax(); // Softmax probabilities
    
    // Print probs
    probs.print();
    
    return 0;
}