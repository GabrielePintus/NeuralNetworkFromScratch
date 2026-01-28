# LAMP: A Neural Network From Scratch in C++

## Overview

**LAMP** is a didactic C++ project that builds a **minimal, PyTorch-like neural network stack** from the ground up.
It focuses on making the core ideas of tensors, autograd, layers, and training loops easy to read and experiment with,
without the complexity of a full production framework.

The name **LAMP** reflects the project’s philosophy: a lightweight, educational library that helps illuminate how
modern deep-learning systems work internally.

## Features

* **Tensor core with autograd**: Multi-dimensional tensors with basic arithmetic, matrix multiplication, reductions,
  and activation functions, plus gradient tracking for learning.
* **Modules and composition**: Layer abstractions (e.g., `Linear`) and a `Sequential` container for chaining modules.
* **Loss functions**: MSE, binary cross-entropy, and cross-entropy with softmax.
* **Optimizers**: SGD and Adam to update model parameters.
* **Data utilities**: Simple datasets and a `DataLoader` for batching and shuffling.


