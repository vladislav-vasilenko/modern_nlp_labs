# Transformer Lab

Minimal PyTorch project for sinusoidal positional embeddings, following the
intuition and formulas from Suvash Sedhain's article
["Positional Encodings for LLMs: From Sinusoidal to RoPE"](https://mesuvash.github.io/blog/2026/positional-encodings/).

The implemented encoding is the original fixed, additive scheme:

```text
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

Token embeddings and positional encodings have the same width, so the transformer
input is just:

```text
x_pos = token_embedding + PE(pos)
```

The project also includes a small rotation helper that checks the relative-position
property described in the article:

```text
PE(pos + k) = M_k PE(pos)
```

where each sine/cosine pair is transformed by a 2D rotation matrix that depends on
the offset `k`, not on the absolute position.

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
uv run --python "$TORCH_RESEARCH_PY" --no-project python examples/positional_lab.py
```

Open the notebook version:

```bash
jupyter notebook notebooks/positional_embeddings_lab.ipynb
```

Open the PyTorch warm-up lab:

```bash
jupyter notebook notebooks/pytorch_shapes_attention_lab.ipynb
```

Run tests:

```bash
uv run --python "$TORCH_RESEARCH_PY" --no-project python -m pytest
```

## Labs

- [PyTorch shapes, dot products, and mini-attention](notebooks/pytorch_shapes_attention_lab.ipynb)
- [Исследование positional embeddings](docs/lab-positional-embeddings.md)
- [Notebook: positional embeddings lab](notebooks/positional_embeddings_lab.ipynb)

## Project Layout

```text
src/transformer_lab/
  positional.py      # sinusoidal PE module and relative shift helper
examples/
  basic_usage.py     # tiny runnable demo
  positional_lab.py  # step-by-step numerical lab
docs/
  lab-positional-embeddings.md
notebooks/
  pytorch_shapes_attention_lab.ipynb
  positional_embeddings_lab.ipynb
tests/
  test_positional.py # formula, shape, and relative-shift tests
```
