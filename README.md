# MoE-SOP: Sewer Overflow Prevention via Multimodal Mixture of Experts LLM

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

This repository contains the implementation code for the paper MoE-SOP: Sewer Overflow Prevention via Multimodal Mixture of Experts Large Language Model.

## 🛠️ Environment Configuration

### System Requirements
- Linux operating system
- Python 3.10

### Creating a Environment
```bash
conda create -n moe-sop python=3.10 -y
conda activate moe-sop
```

### Installation of dependencies
```bash
pip install -r requirements.txt
```

## 🤗 Model Download
1. Install huggingface_hub:
```bash
pip install huggingface_hub
```

2. Download the specified model of the paper:
```bash
huggingface-cli download openai-community/gpt2 --local-dir ./gpt2
...
```
## ⚙️ Parameter Configuration
Before running the main program, update the following path parameters:

### astlingen_main.py
```python
params = {
    'llm_path': './[model-name]/',  # Updated to actual model paths
    # Other parameters...
}
```

### chaohu_mian.py
```python
params = {
    'llm_path': './models/[model-name]',  # Updated to actual model paths
    # Other parameters...
}
```

## 📜 LICENSE
This project is licensed under the [Apache License 2.0](LICENSE).

