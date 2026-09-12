from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# HELPER FILE
# This file stores reusable training and reporting utilities shared by the primitive variants.
# It is not an architecture file. It is also not specific to one recurrent equation.


@dataclass
class PrimitiveSequenceDataset:
    # The corpus is split into train, validation, and test token streams:
    #   D_train = [x_1, ..., x_Ntrain]
    #   D_val   = [x_{Ntrain+1}, ..., x_{Nval}]
    #   D_test  = [x_{Nval+1}, ..., x_N]
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor
    test_tokens: torch.Tensor
    stoi: dict[str, int]
    itos: dict[int, str]


@dataclass
class PrimitiveEpochReport:
    # Stores one epoch-level row for the training report:
    #   R_k = (k, L_train^(k), L_val^(k), L_test^(k))
    epoch: int
    train_loss: float
    val_loss: float
    test_loss: float


def build_sequence_dataset(
    text_path: Path,
    train_fraction: float = 0.9,
    val_fraction: float = 0.05,
) -> PrimitiveSequenceDataset:
    # Reads the raw corpus that defines the token sequence x_1, ..., x_N.
    text = text_path.read_text(encoding="utf-8")
    # Builds the discrete vocabulary V.
    vocab = sorted(set(text))
    # Defines the encoding map char -> integer id.
    stoi = {ch: idx for idx, ch in enumerate(vocab)}
    # Defines the decoding map integer id -> char.
    itos = {idx: ch for ch, idx in stoi.items()}
    # Encodes the full corpus into integer tokens.
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    # Chooses the train/validation/test split points.
    train_end = int(len(encoded) * train_fraction)
    val_end = int(len(encoded) * (train_fraction + val_fraction))
    # Returns D_train, D_val, and D_test.
    return PrimitiveSequenceDataset(
        train_tokens=encoded[:train_end],
        val_tokens=encoded[train_end:val_end],
        test_tokens=encoded[val_end:],
        stoi=stoi,
        itos=itos,
    )


def sample_batch(token_stream: torch.Tensor, seq_len: int, batch_size: int, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    # Randomly samples starting positions i for subsequences:
    #   x^(b) = [x_i, ..., x_{i+T-1}]
    #   y^(b) = [x_{i+1}, ..., x_{i+T}]
    if len(token_stream) <= seq_len:
        raise ValueError("Need at least sequence length + 1 tokens")
    starts = torch.randint(0, len(token_stream) - seq_len, (batch_size,), generator=generator)
    # Collects the input subsequences x_1, ..., x_T for each batch element.
    x = torch.stack([token_stream[i : i + seq_len] for i in starts])
    # Collects the next-token targets shifted by one position.
    y = torch.stack([token_stream[i + 1 : i + seq_len + 1] for i in starts])
    # Returns the supervised pair (inputs, targets).
    return x, y


def next_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #   L = CrossEntropy(z_t, target_t)
    # applied over every batch element and every time step.
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


