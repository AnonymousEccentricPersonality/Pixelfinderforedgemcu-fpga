# Supervised Coordinate Regression for Sparse Images

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Edge AI](https://img.shields.io/badge/Focus-Edge%20AI%20%7C%20Hardware--Software%20Co--design-orange)](#)

A hardware-optimized supervised regression model designed to predict the $(x, y)$ coordinates of a single active pixel (value 255) in a $50 \times 50$ grayscale image. 

While the problem is mathematically simple, this implementation focuses on **Hardware-Software Co-design**, showcasing how deep learning models can be architected specifically for **Microcontrollers (MCUs)** and **FPGAs** using early-exit and parallelization strategies.

---

## 🚀 Architectural Rationale

In edge AI deployment, power and latency are the primary constraints. Since the input is extremely sparse (only 1 out of 2,500 pixels is active), standard full-frame inference is computationally wasteful.



### 1. Microcontroller Strategy (Serial + Early Exit)
* **Architecture:** A two-neuron regression model (one for $X$, one for $Y$).
* **Weights:** Each neuron is mapped to 2,500 unique weights corresponding to the pixel grid.
* **Early Exit:** The model processes pixels as a serial stream. Upon detecting a pixel intensity $> \text{Threshold}$, it immediately triggers the regression calculation for that specific index and terminates the loop. 
* **Efficiency:** Reduces average computational cost by $\approx 50-99\%$ depending on pixel location.
* **Quantization:** Weights are quantized to fixed-point integers to ensure compatibility with low-power MCUs.

### 2. FPGA Strategy (Parallel + Row-wise)
* **Architecture:** 50 parallel processing nodes.
* **Concurrence:** Each node computes one row of the image simultaneously.
* **Hardware Efficiency:** By checking rows in parallel, the FPGA can resolve the coordinate in a fraction of the time required for serial processing.
* **Power Optimization:** The FPGA fabric is designed to halt row-wise operations once the active pixel is identified by any node. Then it exits from the loop and starts inference for next image

---

## 🔮 Future Outlook: SNN Evolution
The ultimate optimization for this sparse problem is a **Spiking Neural Network (SNN)** using **Time-to-First-Spike (TTFS)** encoding. 
* **Zero Dynamic Power:** Dark pixels generate no spikes, resulting in zero dynamic power consumption for 99.96% of the grid.
* **Immediate Resolution:** The single active pixel triggers a spike that immediately resolves the $(x, y)$ coordinate through learned synaptic weights.

---

## 📊 Dataset & Training

* **Data Generation:** 2,500 synthetic images ($50 \times 50$) were generated, each containing exactly one "255" pixel at a unique location.
* **Rationale:** Synthetic data allows for a perfectly balanced dataset, ensuring the model treats all $(x, y)$ coordinates with equal priority.
* **Loss Function:** Mean Squared Error (MSE), optimizing for precise coordinate localization.

---

## 💻 Installation & Usage

### Dependencies
Ensure you have Python 3.8+ installed. The project requires:
* `numpy`
* `matplotlib`
* `tensorflow` or `pytorch`

### Setup
```bash
# Clone the repository
git clone <repository-url>

# Navigate to the directory
cd supervised-coordinate-regression

# Install requirements
pip install numpy matplotlib tensorflow
