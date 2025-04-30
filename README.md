# Neural Network lib

This repository provides a modular C++/CUDA neural network framework for research and experimentation. It supports both CPU and GPU backends, custom layers, and flexible data loading. The codebase is designed for extensibility, educational use, and benchmarking.

## Features

*   Modular neural network architecture (CPU & CUDA support)
*   Layer abstraction (Linear, Conv2D, etc.)
*   Customizable optimizers (Adam, SGD, etc.)
*   DataLoader utilities for datasets (MNIST, CIFAR-10, Boston, California Housing, etc.)
*   Debugging and inspection utilities for tensors and gradients
*   Model serialization (save/load parameters)
*   Utilities for memory management and performance monitoring

## Core Components

*   `NeuralNetwork` class: Build, train, and evaluate neural networks.
*   `Layer` classes: Implement various neural network layers.
*   `DataLoader`: Efficient batch loading for datasets.
*   `optimizer`: Optimizer implementations (Adam, SGD, etc.)
*   `debug` and `DebugUtils`: Debugging, logging, and inspection tools.

## How to Use
## IMPORTANT:
Datasets must be placed inside a /Dataset folder in root dir.



### 1. Build the Project

```sh
# Build everything
make

# Build only the library and necessary components
make library

