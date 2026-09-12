from __future__ import annotations
import torch
from torch import nn
from primitive_peephole_lstm import PrimitivePeepholeLSTMBackbone

class PrimitiveCharPeepholeLSTMLanguageModel(nn.Module):
    # Equation set (B1) in primitive_peephole_lstm_usage.md:
    #   e_t = Embedding(x_t)
    #   h_t = PrimitivePeepholeLSTM(e_t, h_{t-1}, c_{t-1})
    #   z_t = h_t W_hy + b_y
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.backbone = PrimitivePeepholeLSTMBackbone(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        hidden_sequence, _ = self.backbone(embedded)
        logits = self.output(hidden_sequence)
        return logits
