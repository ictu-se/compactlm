from __future__ import annotations

import torch
from torch import nn

# ARCHITECTURE FILE
# This file contains only the primitive peephole LSTM equations and their direct implementation.
# It does not contain data loading, optimization, early stopping, or text generation policy.


class PrimitivePeepholeLSTMCell(nn.Module):
    # Equation set (A1) in primitive_peephole_lstm.md:
    #   f_t = sigmoid(x_t W_xf + h_{t-1} W_hf + c_{t-1} ⊙ p_f + b_f)
    #   i_t = sigmoid(x_t W_xi + h_{t-1} W_hi + c_{t-1} ⊙ p_i + b_i)
    #   g_t = tanh(x_t W_xg + h_{t-1} W_hg + b_g)
    #   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    #   o_t = sigmoid(x_t W_xo + h_{t-1} W_ho + c_t ⊙ p_o + b_o)
    #   h_t = o_t ⊙ tanh(c_t)
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_xf = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_hf = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.p_f = nn.Parameter(torch.zeros(hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))
        self.W_xi = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_hi = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.p_i = nn.Parameter(torch.zeros(hidden_dim))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        self.W_xg = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_hg = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))
        self.W_xo = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_ho = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.p_o = nn.Parameter(torch.zeros(hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_xf)
        nn.init.orthogonal_(self.W_hf)
        nn.init.xavier_uniform_(self.W_xi)
        nn.init.orthogonal_(self.W_hi)
        nn.init.xavier_uniform_(self.W_xg)
        nn.init.orthogonal_(self.W_hg)
        nn.init.xavier_uniform_(self.W_xo)
        nn.init.orthogonal_(self.W_ho)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_t = torch.sigmoid((x_t @ self.W_xf) + (h_prev @ self.W_hf) + (c_prev * self.p_f) + self.b_f)
        i_t = torch.sigmoid((x_t @ self.W_xi) + (h_prev @ self.W_hi) + (c_prev * self.p_i) + self.b_i)
        g_t = torch.tanh((x_t @ self.W_xg) + (h_prev @ self.W_hg) + self.b_g)
        c_t = (f_t * c_prev) + (i_t * g_t)
        o_t = torch.sigmoid((x_t @ self.W_xo) + (h_prev @ self.W_ho) + (c_t * self.p_o) + self.b_o)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class PrimitivePeepholeLSTMBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = PrimitivePeepholeLSTMCell(input_dim, hidden_dim)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        return h0, c0

    def forward(self, x: torch.Tensor, state0: tuple[torch.Tensor, torch.Tensor] | None = None):
        batch_size, seq_len, _ = x.shape
        if state0 is None:
            h_t, c_t = self.initial_state(batch_size, x.device, x.dtype)
        else:
            h_t, c_t = state0
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, h_t, c_t)
            outputs.append(h_t)
        H = torch.stack(outputs, dim=1)
        return H, (h_t, c_t)
