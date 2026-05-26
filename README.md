# visAI — Real-Time PyTorch Training Visualizer

Monitor your PyTorch model's training process live in the browser — no extra setup, just a few function calls.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)

---

## Features

- **Loss Tracking** — Train and validation loss plotted in real time
- **Class-wise Accuracy** — Side-by-side train vs. validation accuracy per class
- **Weight Heatmap** — Visualize the weight matrix of any layer across epochs
- **Activation Histogram** — Distribution of neuron activations during training
- **Gradient Norm** — Track gradient norms to detect vanishing/exploding gradients
- **Training Controls** — Start, Stop, and Resume training from the browser dashboard

---

## Demo

<img width="1451" height="696" alt="image" src="https://github.com/user-attachments/assets/6dc58fa0-6d7e-4c0e-bba0-6d554d47bf1d" />

---

## How It Works

```
Your training script  →  HTTP POST  →  visAI server (FastAPI)  →  WebSocket  →  Browser dashboard
```

`visAI.connect()` spins up a local FastAPI server and opens your browser automatically. Each `get_*` call sends data to the server, which streams it to the dashboard via WebSocket in real time.

---

## Installation

Install the required dependencies:

```bash
pip install fastapi uvicorn websockets requests torch scikit-learn numpy pandas
```

Copy `visAI.py` into your project directory — no package installation needed.

---

## Quick Start

```python
import visAI

# 1. Start the server and open the browser
visAI.connect('127.0.0.1')
visAI.get_log('Training started')

# 2. Inside your training loop
for epoch in range(n_epochs):
    # ... training and validation steps ...

    visAI.get_loss(n_epochs, epoch, train_loss, valid_loss)
    visAI.get_acc(n_epochs, epoch, y_hat_train, y_train, y_hat_valid, y_valid)
    visAI.get_weight(n_epochs, epoch, model.fc3.weight)
    visAI.get_act(n_epochs, epoch, activation_output)
    visAI.get_gradnorm(n_epochs, epoch, grad_norm)

# 3. Shut down the server
visAI.get_log('END')
visAI.disconnect()
```

Open your browser at **http://127.0.0.1:8000** — it launches automatically when you run the script.

---

## API Reference

| Function | Arguments | Description |
|---|---|---|
| `connect(host)` | `host: str` | Starts the FastAPI server and opens the dashboard in the browser. Blocks until you click **Start** in the UI. |
| `disconnect()` | — | Shuts down the server. |
| `get_log(message)` | `message: str` | Sends a log message visible in the browser console. |
| `get_loss(n_epochs, epoch, train_loss, valid_loss)` | `float, float` | Updates the loss line chart. |
| `get_acc(n_epochs, epoch, y_hat_train, y_train, y_hat_valid, y_valid)` | array-like | Computes and sends class-wise accuracy for train and validation sets. |
| `get_weight(n_epochs, epoch, weight_tensor)` | `torch.Tensor` | Sends the weight matrix of a layer as a heatmap. Automatically downsamples large matrices to 15×15. |
| `get_act(n_epochs, epoch, activation_tensor)` | `torch.Tensor` | Sends neuron activation values for histogram display. |
| `get_gradnorm(n_epochs, epoch, grad_norm)` | `float` | Sends the gradient norm (L2) of a layer for trend visualization. |

---

## Dashboard Controls

| Button | Action |
|---|---|
| **Start** | Signals the training script to begin. Must be clicked before training proceeds. |
| **Stop** | Pauses the data stream to the dashboard (training continues in background). |
| **Resume** | Resumes the data stream. |

---

## Examples

Two complete working examples are included:

**`AI_model_BC.py`** — Binary classification on the Breast Cancer Wisconsin dataset (sklearn)
- 6-layer MLP with LeakyReLU activations
- Binary cross-entropy loss, Adam optimizer
- Early stopping with best-model checkpointing

**`Mnist.py`** — Multi-class classification on MNIST handwritten digits (torchvision)
- 7-layer MLP with LeakyReLU + LogSoftmax
- NLL loss, Adam optimizer
- GPU support via `torch.device('cuda')` if available

---

## Project Structure

```
.
├── visAI.py            # Core library — server, WebSocket, and visualization API
├── AI_model_BC.py      # Example: Breast Cancer binary classification
├── Mnist.py            # Example: MNIST multi-class classification
├── index.html          # Dashboard UI (Vega-Lite charts)
├── static/
│   ├── script.js       # WebSocket client and chart rendering
│   └── style.css       # Dashboard styles
└── assets/
    └── demo.png         # Full dashboard screenshot or GIF for README
```

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` | Model training |
| `fastapi` + `uvicorn` | Local HTTP/WebSocket server |
| `websockets` | WebSocket client support |
| `requests` | Sending data from training script to server |
| `scikit-learn` | Confusion matrix for accuracy computation |
| `numpy` + `pandas` | Data processing |
| `vega` + `vega-lite` + `vega-embed` | Chart rendering (loaded via CDN in browser) |

---
