# 🔥 FTRAIN V1.0

> High-performance, lightweight AI training framework engineered for maximum speed, memory efficiency, and scalable LLM workflows.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## 🚀 Key Features

* **High-Throughput Execution:** Low-overhead data loaders and optimized training loops designed to eliminate GPU starvation.
* **Memory Efficient:** Optimized VRAM usage to enable fine-tuning and pre-training on accessible hardware.
* **Flexible Scaling:** Seamless execution across small parameters (0.5B) up to larger foundation models (7B+).
* **Developer-Centric:** Modular architecture built for rapid prototyping and production training.

---

## 📊 Benchmarks

Preliminary performance benchmarks comparing **0.5B** and **7B** model training workflows:

| Model Scale | Sequence Length | VRAM Usage (Avg) | Throughput | Training Loss |
| :--- | :--- | :--- | :--- | :--- |
| **0.5B Model** | 2048 | ~4.5 GB | ~18,500 tok/s | Fast convergence |
| **7B Model** | 2048 | ~16.8 GB | ~4,200 tok/s | Stable scaling |

> *Note: Benchmarks measured on standard CUDA/PyTorch environments. Replace sample VRAM/Throughput numbers above with your exact hardware test results.*

---

## 🛠️ Installation & Getting Started

PyPI package distribution is coming soon! In the meantime, you can build and install directly from source:

### 1. Clone the Repository
```bash
git clone [https://github.com/aiphoenixlabs/Ftrain.git](https://github.com/aiphoenixlabs/Ftrain.git)
cd Ftrain
