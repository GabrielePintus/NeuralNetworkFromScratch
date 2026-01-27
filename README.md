# LAMP: A Neural Network From Scratch in C++

## Overview

**LAMP** is a didactic project whose goal is to implement the *simplest possible version of PyTorch* in C++.  
It is designed as a learning exercise to understand how tensors, automatic differentiation (autograd), and neural
network modules work internally.

The name **LAMP** reflects this philosophy: a *lightweight* and *minimal* PyTorch-like framework built for
educational purposes.

## Scope

This repository is a compact C++ learning project that builds a minimal neural
network stack from scratch. It focuses on implementing core tensor operations,
autograd, and a small module system, along with example programs that exercise
the components end-to-end.

### In scope
- Tensor math and autograd basics.
- Simple neural network modules (e.g., linear layers) and composition helpers.
- Lightweight training utilities and examples to demonstrate learning loops.

### Out of scope
- Full deep learning framework features (GPU, large model zoo, advanced
  optimizers, serialization, etc.).
- Production-ready performance or extensive API stability guarantees.