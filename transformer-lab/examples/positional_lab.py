"""Step-by-step numerical lab for sinusoidal positional embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformer_lab import (
    SinusoidalPositionalEncoding,
    make_sinusoidal_positional_encoding,
    shift_sinusoidal_encoding,
)


def print_block(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def rounded(values: Tensor, decimals: int = 4) -> list[float]:
    return values.detach().cpu().round(decimals=decimals).tolist()


def step_1_order_problem() -> None:
    print_block("1. Adding position vectors")
    torch.manual_seed(7)

    batch_size = 2
    seq_len = 6
    d_model = 8

    token_embeddings = torch.randn(batch_size, seq_len, d_model)
    positional = SinusoidalPositionalEncoding(d_model, max_len=32, dropout=0.0)
    position_aware = positional(token_embeddings)

    print("token embeddings shape:    ", tuple(token_embeddings.shape))
    print("position-aware shape:      ", tuple(position_aware.shape))
    print("first token before PE:      ", rounded(token_embeddings[0, 0]))
    print("first token after PE:       ", rounded(position_aware[0, 0]))
    print("PE(0):                      ", rounded(positional.pe[0, 0]))


def step_2_formula_probe() -> None:
    print_block("2. Sinusoidal formula probe")

    max_len = 8
    d_model = 8
    pe = make_sinusoidal_positional_encoding(max_len=max_len, d_model=d_model)

    print("PE matrix shape:            ", tuple(pe.shape))
    for pos in range(4):
        print(f"PE({pos}):                     ", rounded(pe[pos]))


def step_3_similarity_by_distance() -> None:
    print_block("3. Similarity by distance")

    max_len = 128
    d_model = 32
    base = 10_000.0
    pe = make_sinusoidal_positional_encoding(max_len=max_len, d_model=d_model, base=base)

    for distance in [1, 2, 4, 8, 16, 32]:
        left = pe[:-distance]
        right = pe[distance:]
        similarity = F.cosine_similarity(left, right, dim=-1).mean().item()
        print(f"distance={distance:>2}: mean cosine similarity={similarity:.4f}")


def step_4_relative_shift() -> None:
    print_block("4. Relative shift property")

    max_len = 64
    d_model = 16
    offset = 7
    pe = make_sinusoidal_positional_encoding(max_len=max_len, d_model=d_model)
    shifted = shift_sinusoidal_encoding(pe[:-offset], offset)
    max_error = (shifted - pe[offset:]).abs().max().item()

    print("offset:                     ", offset)
    print(f"relative shift max error:    {max_error:.2e}")
    print("PE(0) shifted by offset:     ", rounded(shifted[0]))
    print("PE(offset):                 ", rounded(pe[offset]))


def step_5_module_start_pos() -> None:
    print_block("5. Module start_pos check")

    d_model = 8
    seq_len = 3
    start_pos = 4
    positional = SinusoidalPositionalEncoding(d_model=d_model, max_len=16, dropout=0.0)
    x = torch.zeros(1, seq_len, d_model)

    out = positional(x, start_pos=start_pos)
    expected = positional.pe[:, start_pos : start_pos + seq_len, :]
    max_error = (out - expected).abs().max().item()

    print("input shape:                 ", tuple(x.shape))
    print("selected PE shape:           ", tuple(expected.shape))
    print(f"start_pos max error:         {max_error:.2e}")


def main() -> None:
    step_1_order_problem()
    step_2_formula_probe()
    step_3_similarity_by_distance()
    step_4_relative_shift()
    step_5_module_start_pos()


if __name__ == "__main__":
    main()
