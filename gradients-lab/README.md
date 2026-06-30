# Gradients Lab

Minimal PyTorch project for experiments and labs on deep learning gradients.

## Labs

- [Лабораторная работа: исследование градиентов](docs/lab-gradients.md)
- [Notebook: gradients lab](notebooks/gradients_lab.ipynb)

## Setup

This project is set up for `uv` while using the existing conda environment:

```bash
export TORCH_RESEARCH_PY=/Users/vladmac/Code/dl/miniconda3/envs/torch-research/bin/python
uv pip install --python "$TORCH_RESEARCH_PY" -e ".[dev]"
```

Run the example:

```bash
uv run --python "$TORCH_RESEARCH_PY" --no-project python examples/basic_usage.py
```

Run the step-by-step lab:

```bash
uv run --python "$TORCH_RESEARCH_PY" --no-project python examples/gradients_lab.py
```

Open the notebook version:

```bash
jupyter notebook notebooks/gradients_lab.ipynb
```

Run tests:

```bash
uv run --python "$TORCH_RESEARCH_PY" --no-project python -m pytest
```
