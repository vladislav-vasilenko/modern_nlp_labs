from __future__ import annotations

import math

import pytest
import torch

from transformer_lab import (
    SinusoidalPositionalEncoding,
    make_sinusoidal_positional_encoding,
    shift_sinusoidal_encoding,
)


def test_encoding_matches_formula() -> None:
    pe = make_sinusoidal_positional_encoding(max_len=4, d_model=6)

    pos = 3
    pair_index = 1
    denominator = 10_000 ** (2 * pair_index / 6)
    angle = pos / denominator

    assert pe[pos, 2 * pair_index].item() == pytest.approx(math.sin(angle))
    assert pe[pos, 2 * pair_index + 1].item() == pytest.approx(math.cos(angle))


def test_module_adds_positions_batch_first() -> None:
    positional = SinusoidalPositionalEncoding(d_model=8, max_len=10, dropout=0.0)
    x = torch.zeros(2, 4, 8)

    out = positional(x)

    assert out.shape == x.shape
    torch.testing.assert_close(out[0], positional.pe[0, :4])
    torch.testing.assert_close(out[1], positional.pe[0, :4])


def test_module_supports_start_pos() -> None:
    positional = SinusoidalPositionalEncoding(d_model=8, max_len=10, dropout=0.0)
    x = torch.zeros(1, 3, 8)

    out = positional(x, start_pos=4)

    torch.testing.assert_close(out[0], positional.pe[0, 4:7])


def test_relative_shift_property() -> None:
    pe = make_sinusoidal_positional_encoding(max_len=20, d_model=16)
    offset = 5

    shifted = shift_sinusoidal_encoding(pe[:-offset], offset)

    torch.testing.assert_close(shifted, pe[offset:], atol=1e-6, rtol=1e-6)


def test_d_model_must_be_even() -> None:
    with pytest.raises(ValueError, match="even"):
        make_sinusoidal_positional_encoding(max_len=4, d_model=7)
