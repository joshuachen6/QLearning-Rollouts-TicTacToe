# Q-Learning with ResNet Rollouts for TicTacToe

This project implements a reinforcement learning agent for TicTacToe using Q-Learning and a ResNet architecture. It leverages LibTorch for neural network operations and provides a graphical interface for training and configuration.

## Overview

The primary goal of this project is to explore the application of deep reinforcement learning to simple state-space games. While TicTacToe is traditionally solvable with simpler methods, this implementation serves as a learning platform for:
- Integrating LibTorch with C++ for deep learning.
- Implementing ResNet architectures in a high-performance environment.
- Managing large state-action spaces with SQLite-backed Q-tables.
- Building interactive training monitoring tools with ImGui and SFML.

## Current Status

This project is currently a work in progress. It is intended as a pedagogical exercise rather than a functional competitive agent. Note that the ResNet architecture may be overly complex for the limited data provided by a 3x3 TicTacToe board, leading to potential convergence challenges.

## Key Features

- **ResNet Architecture**: A deep residual network implemented via LibTorch to approximate value functions.
- **Persistent Q-Table**: State-action values are managed using an SQLite3 database for scalability and persistence across sessions.
- **Multi-threaded Training**: Support for parallel board simulations and training using OpenMP.
- **Graphical Interface**: A comprehensive UI built with ImGui and SFML for real-time training monitoring and parameter configuration.
- **Configuration Management**: External TOML-based configuration for easy adjustment of hyperparameters like epsilon, alpha, and batch size.

## Technical Stack

- **Languages**: C++17
- **Deep Learning**: LibTorch (PyTorch C++ API)
- **GUI**: ImGui, SFML
- **Database**: SQLite3
- **Build System**: xmake
- **Data Format**: TOML

## Building and Running

### Prerequisites

- [xmake](https://xmake.io/)
- A C++17 compatible compiler (GCC, Clang, or MSVC)
- Dependencies (automatically handled by xmake): SQLite3, toml11, OpenMP, SFML, ImGui

### Build Instructions

1. Clone the repository.
2. Build the project using xmake:
   ```bash
   xmake
   ```
   Note: On the first build, xmake will automatically download the appropriate LibTorch distribution for your platform.

3. Run the application:
   ```bash
   xmake run TicTacToe
   ```

## License

This project is open-source and intended for educational purposes.
