# LAMP: A Neural Network From Scratch in C++

<img src="Lamp_logo.png" alt="Lamp logo" width="300"/>

## Overview

**LAMP** is a didactic C++ project that builds a **minimal, PyTorch-like neural network stack** from the ground up.
It focuses on making the core ideas of tensors, autograd, layers, and training loops easy to read and experiment with,
without the complexity of a full production framework.

## Why this project exists

LAMP is designed for people who want to understand how neural networks work *under the hood*:

- Learn how tensors are represented and manipulated.
- Explore automatic differentiation by reading a compact autograd engine.
- See how layers, losses, and optimizers connect in a training loop.
- Experiment with a tiny but coherent deep-learning stack that you can extend.

It is not intended to compete with full frameworks like PyTorch or TensorFlow.

## Features

- **Tensor core with autograd**: Multi-dimensional tensors with basic arithmetic, matrix multiplication,
  reductions, and activation functions, plus gradient tracking for learning.
- **Modules and composition**: Layer abstractions (e.g., `Linear`) and a `Sequential` container for chaining modules.
- **Loss functions**: MSE, binary cross-entropy, and cross-entropy.
- **Optimizers**: SGD and Adam to update model parameters.
- **Data utilities**: Simple datasets and a `DataLoader` for batching and shuffling.

## Repository layout

```text
.
├── CMakeLists.txt         # Build configuration
├── include/               # Public headers (core API)
├── src/                   # Implementation files
└── examples/              # Example training runs
```

## Build instructions

LAMP uses CMake and a standard out-of-source build.

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The flag `-DCMAKE_BUILD_TYPE=Release` enables optimizations for better performance. You can also use `Debug` for easier debugging.

## Running examples

After building, run the examples from the `build` directory. For example:

```bash
./build/examples/<example_binary_name>
```

Check the `examples/` folder to see what programs are available and what they demonstrate (e.g., linear regression,
classification, or toy datasets).

## Example: training loop

Below is a high-level sketch of what the API is designed to feel like. The exact API may differ depending on the
example you use, but the structure should be familiar if you’ve used PyTorch.

```cpp
Tensor x = ...; // input batch
Tensor y = ...; // labels

auto model = nn::Sequential({
    std::make_shared<nn::Linear>(input_dim, hidden_dim),
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::Linear>(hidden_dim, output_dim)
});

auto optimizer = nn::Adam(model.parameters(), /*lr=*/1e-3);

for (int epoch = 0; epoch < epochs; ++epoch) {
    Tensor preds = model.forward(x);
    Tensor loss = cross_entropy(preds, y);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```


## License

This project is licensed under the terms of the **LICENSE** file in the repository.
