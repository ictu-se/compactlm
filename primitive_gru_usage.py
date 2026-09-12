from __future__ import annotations
import torch
from torch import nn
from primitive_gru import PrimitiveGRUBackbone

class PrimitiveCharGRULanguageModel(nn.Module):
    # Equation set (B1) in primitive_gru_usage.md:
    #   e_t = Embedding(x_t)
    #   h_t = PrimitiveGRU(e_t, h_{t-1})
    #   z_t = h_t W_hy + b_y
    #   p(x_{t+1} | x_{<=t}) = softmax(z_t)
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()  # Standard nn.Module initialization.
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Equation (B1): e_t = Embedding(x_t).
        self.backbone = PrimitiveGRUBackbone(embed_dim, hidden_dim)  # Equation (B1): h_t = PrimitiveGRU(e_t, h_{t-1}).
        self.output = nn.Linear(hidden_dim, vocab_size)  # Equation (B1): z_t = h_t W_hy + b_y.

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Equation (B1): convert token ids x_t into embedding vectors e_t.
        embedded = self.embedding(token_ids)
        # Equation (B1): run the primitive recurrent update through time to get h_1, ..., h_T.
        hidden_sequence, _ = self.backbone(embedded)
        # Equation (B1): project each h_t into next-token logits z_t.
        logits = self.output(hidden_sequence)
        # Returns z_t for all time steps.
        return logits
