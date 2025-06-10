# Chess GPT Dashboard

A web-based visualization dashboard for analyzing the behavior of a GPT-2 language model trained on chess UCI move sequences. Built using Dash, Plotly, and PyTorch, the app enables interactive inspection of transformer predictions, attention patterns, and move dynamics.

## Features

- Visualize chess boards using `python-chess` and SVG rendering
- Interactive UI built with Dash and Bootstrap components
- Load transformer models from Hugging Face
- Support for GPU-accelerated inference via PyTorch (CUDA 12.6)
- Plot residuals, hidden states, or model outputs with Plotly

## Installation

### Requirements

- Python 3.11.11
- pip with CUDA-enabled PyTorch source

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chess-gpt-dashboard.git
   cd chess-gpt-dashboard
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
   ```

## Usage

1. Launch the dashboard:

   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:8050`