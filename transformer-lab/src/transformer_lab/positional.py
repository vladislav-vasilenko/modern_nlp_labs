"""Sinusoidal positional embeddings for transformer inputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _check_even_d_model(d_model: int) -> None:
    if d_model <= 0:
        raise ValueError("d_model must be positive.")
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for paired sine/cosine dimensions.")


def sinusoidal_frequencies(
    d_model: int,
    *,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return frequencies omega_i = 1 / base^(2i / d_model)."""

    _check_even_d_model(d_model)
    pair_indices = torch.arange(0, d_model, 2, device=device, dtype=dtype)
    return torch.exp(-torch.log(torch.tensor(base, device=device, dtype=dtype)) * pair_indices / d_model)


def make_sinusoidal_positional_encoding(
    max_len: int,
    d_model: int,
    *,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a fixed positional encoding matrix of shape ``(max_len, d_model)``."""

    if max_len <= 0:
        raise ValueError("max_len must be positive.")

    frequencies = sinusoidal_frequencies(d_model, base=base, device=device, dtype=dtype)
    positions = torch.arange(max_len, device=device, dtype=dtype).unsqueeze(1)
    angles = positions * frequencies.unsqueeze(0)

    pe = torch.empty(max_len, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


def shift_sinusoidal_encoding(
    encoding: Tensor,
    offset: int | float | Tensor,
    *,
    base: float = 10_000.0,
) -> Tensor:
    """Apply the article's pairwise rotation matrix to get ``PE(pos + offset)``.

    The input must be a sinusoidal encoding tensor whose last dimension is
    ``d_model``. The returned tensor has the same shape.
    """

    d_model = encoding.shape[-1]
    _check_even_d_model(d_model)

    frequencies = sinusoidal_frequencies(
        d_model,
        base=base,
        device=encoding.device,
        dtype=encoding.dtype,
    )
    offset_tensor = torch.as_tensor(offset, device=encoding.device, dtype=encoding.dtype)
    angles = offset_tensor * frequencies
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    even = encoding[..., 0::2]
    odd = encoding[..., 1::2]

    shifted = torch.empty_like(encoding)
    shifted[..., 0::2] = cos * even + sin * odd
    shifted[..., 1::2] = -sin * even + cos * odd
    return shifted


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position vectors to token embeddings.

    Args:
        d_model: Token embedding size. Must be even.
        max_len: Number of positions to precompute.
        dropout: Dropout applied after adding positions.
        batch_first: If true, inputs are shaped ``(batch, seq, d_model)``.
        base: Frequency base. The original transformer uses 10,000.
    """

    def __init__(
        self,
        d_model: int,
        *,
        max_len: int = 5_000,
        dropout: float = 0.0,
        batch_first: bool = True,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        pe = make_sinusoidal_positional_encoding(max_len, d_model, base=base).unsqueeze(0)
        if not batch_first:
            pe = pe.transpose(0, 1)

        self.d_model = d_model
        self.max_len = max_len
        self.batch_first = batch_first
        self.base = base
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor, *, start_pos: int = 0) -> Tensor:
        """Add positional encodings to ``x``.

        ``start_pos`` is useful during autoregressive decoding when the current
        chunk starts after already-processed tokens.
        """

        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        end_pos = start_pos + seq_len
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative.")
        if end_pos > self.max_len:
            raise ValueError(
                f"Need positions up to {end_pos}, but max_len is {self.max_len}."
            )

        if self.batch_first:
            positions = self.pe[:, start_pos:end_pos, :]
        else:
            positions = self.pe[start_pos:end_pos, :, :]
        return self.dropout(x + positions)
