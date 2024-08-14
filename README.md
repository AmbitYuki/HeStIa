# HESTIA: Asynchronous Embodied Dynamic Locomotion Learning for Diverse Walking Robots through Multimodal Large Language Models

This repository contains the implementation of HESTIA (Heuristic Embodied Simulation for Task-driven Intelligent Adaptation), a novel framework for asynchronous embodied dynamic locomotion learning in diverse walking robots using multimodal large language models.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Docker Support](#docker-support)
6. [Contributing](#contributing)


## Introduction

HESTIA is a cutting-edge framework that integrates multimodal large language models (MLLMs) with reinforcement learning techniques to enhance locomotion learning in robotic systems. This project implements the architecture and methodologies described in our paper [insert paper link or reference].

## Installation

To set up the HESTIA framework, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/HESTIA.git
   cd HESTIA
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

The repository is organized as follows:

- `envs/`: Contains environment configurations and simulation settings
- `network/`: Implements neural network architectures used in HESTIA
- `rl/`: Reinforcement learning algorithms and PPO implementation
- `scripts/`: Utility scripts for data processing and visualization
- `tasks/`: Defines specific locomotion tasks and challenges
- `util/`: Utility functions and helper modules
- `LLAVA_output/`: Stores outputs from the LLAVA model
- `docker_run.sh`: Script for running the project in a Docker container

## Usage

To run a basic locomotion learning experiment:

```
python main.py --task bipedal_walk --model hestia --epochs 1000
```

For more detailed options and configurations, refer to the documentation in each module.

## Docker Support

We provide Docker support for easy deployment and reproducibility. To run HESTIA using Docker:

1. Build the Docker image:
   ```
   docker build -t hestia:latest .
   ```

2. Run the container:
   ```
   ./docker_run.sh
   ```

This will start an interactive session within the Docker container with all necessary dependencies installed.

## Contributing

For any questions or issues, please open an issue on the GitHub repository or contact the authors directly.
