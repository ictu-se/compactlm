from __future__ import annotations

import torch
from torch import nn

# ARCHITECTURE FILE
# This file contains only the primitive CIFG LSTM equations and their direct implementation.
# It does not contain data loading, optimization, early stopping, or text generation policy.


class PrimitiveCIFGLSTMCell(nn.Module):
    # Equation set (A1) in primitive_cifg_lstm.md:
    #   f_t = sigmoid(x_t W_xf + h_{t-1} W_hf + b_f)
    #   i_t = 1 - f_t
    #   g_t = tanh(x_t W_xg + h_{t-1} W_hg + b_g)
    #   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    #   o_t = sigmoid(x_t W_xo + h_{t-1} W_ho + b_o)
    #   h_t = o_t ⊙ tanh(c_t)
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_xf = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_hf = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))
        self.W_xg = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_hg = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))
        self.W_xo = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_ho = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_xf)
        nn.init.orthogonal_(self.W_hf)
        nn.init.xavier_uniform_(self.W_xg)
        nn.init.orthogonal_(self.W_hg)
        nn.init.xavier_uniform_(self.W_xo)
        nn.init.orthogonal_(self.W_ho)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_t = torch.sigmoid((x_t @ self.W_xf) + (h_prev @ self.W_hf) + self.b_f)
        i_t = 1.0 - f_t
        g_t = torch.tanh((x_t @ self.W_xg) + (h_prev @ self.W_hg) + self.b_g)
        c_t = (f_t * c_prev) + (i_t * g_t)
        o_t = torch.sigmoid((x_t @ self.W_xo) + (h_prev @ self.W_ho) + self.b_o)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class PrimitiveCIFGLSTMBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = PrimitiveCIFGLSTMCell(input_dim, hidden_dim)

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
