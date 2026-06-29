"""Tiny demo of additive sinusoidal positional embeddings."""

from __future__ import annotations

import torch

from transformer_lab import (
    SinusoidalPositionalEncoding,
    make_sinusoidal_positional_encoding,
    shift_sinusoidal_encoding,
)


def main() -> None:
    torch.manual_seed(7)

    batch_size = 2
    seq_len = 6
    d_model = 8

    token_embeddings = torch.randn(batch_size, seq_len, d_model)
    positional = SinusoidalPositionalEncoding(d_model, max_len=32, dropout=0.0)
    position_aware = positional(token_embeddings)

    print("input shape:          ", tuple(token_embeddings.shape))
    print("position-aware shape: ", tuple(position_aware.shape))
    print("first PE vector:      ", positional.pe[0, 0].round(decimals=4).tolist())

    pe = make_sinusoidal_positional_encoding(max_len=16, d_model=d_model)
    offset = 3
    shifted = shift_sinusoidal_encoding(pe[:-offset], offset)
    max_error = (shifted - pe[offset:]).abs().max().item()
    print(f"relative shift error: {max_error:.2e}")


if __name__ == "__main__":
    main()
