from __future__ import annotations

import torch
from torch import nn

# ARCHITECTURE FILE
# This file contains only the primitive GRU equations and their direct implementation.
# It does not contain data loading, optimization, early stopping, or text generation policy.


class PrimitiveGRUCell(nn.Module):
    # Equation set (A1) in primitive_gru.md:
    #   z_t = sigmoid(x_t W_xz + h_{t-1} W_hz + b_z)
    #   r_t = sigmoid(x_t W_xr + h_{t-1} W_hr + b_r)
    #   h~_t = tanh(x_t W_xh + (r_t ⊙ h_{t-1}) W_hh + b_h)
    #   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()  # Standard nn.Module initialization; not part of the math itself.
        self.input_dim = input_dim  # Stores d_x, the width of x_t in Equation set (A1).
        self.hidden_dim = hidden_dim  # Stores d_h, the width of h_t in Equation set (A1).

        # Equation (A1): W_xz in R^{d_x x d_h}.
        self.W_xz = nn.Parameter(torch.empty(input_dim, hidden_dim))
        # Equation (A1): W_hz in R^{d_h x d_h}.
        self.W_hz = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        # Equation (A1): b_z in R^{d_h}.
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))

        # Equation (A1): W_xr in R^{d_x x d_h}.
        self.W_xr = nn.Parameter(torch.empty(input_dim, hidden_dim))
        # Equation (A1): W_hr in R^{d_h x d_h}.
        self.W_hr = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        # Equation (A1): b_r in R^{d_h}.
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))

        # Equation (A1): W_xh in R^{d_x x d_h}.
        self.W_xh = nn.Parameter(torch.empty(input_dim, hidden_dim))
        # Equation (A1): W_hh in R^{d_h x d_h}.
        self.W_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        # Equation (A1): b_h in R^{d_h}.
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        self.reset_parameters()  # Chooses initial numeric values for the symbols in Equation set (A1).

    def reset_parameters(self) -> None:
        # This is initialization only; it is not one of the forward equations in the markdown.
        nn.init.xavier_uniform_(self.W_xz)
        nn.init.orthogonal_(self.W_hz)
        nn.init.zeros_(self.b_z)
        nn.init.xavier_uniform_(self.W_xr)
        nn.init.orthogonal_(self.W_hr)
        nn.init.zeros_(self.b_r)
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.orthogonal_(self.W_hh)
        nn.init.zeros_(self.b_h)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Equation (A1): z_t = sigmoid(x_t W_xz + h_{t-1} W_hz + b_z).
        z_t = torch.sigmoid((x_t @ self.W_xz) + (h_prev @ self.W_hz) + self.b_z)
        # Equation (A1): r_t = sigmoid(x_t W_xr + h_{t-1} W_hr + b_r).
        r_t = torch.sigmoid((x_t @ self.W_xr) + (h_prev @ self.W_hr) + self.b_r)
        # Equation (A1): r_t ⊙ h_{t-1}.
        reset_hidden = r_t * h_prev
        # Equation (A1): h~_t = tanh(x_t W_xh + (r_t ⊙ h_{t-1}) W_hh + b_h).
        h_tilde = torch.tanh((x_t @ self.W_xh) + (reset_hidden @ self.W_hh) + self.b_h)
        # Equation (A1): h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t.
        h_t = ((1.0 - z_t) * h_prev) + (z_t * h_tilde)
        # Returns h_t from Equation set (A1).
        return h_t


class PrimitiveGRUBackbone(nn.Module):
    # Equation set (A2) in primitive_gru.md:
    #   h_0 = 0
    #   h_t = GRUCell(x_t, h_{t-1}),  t = 1, ..., T
    #   H = [h_1, h_2, ..., h_T]
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()  # Standard nn.Module initialization.
        self.input_dim = input_dim  # Stores d_x from Equation (A2).
        self.hidden_dim = hidden_dim  # Stores d_h from Equation (A2).
        self.cell = PrimitiveGRUCell(input_dim, hidden_dim)  # Uses Equation set (A1) at each time step.

    def initial_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Equation (A2): h_0 = 0.
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x has shape [B, T, d_x], which is the batch of sequences x_1, ..., x_T from Equation (A2).
        batch_size, seq_len, _ = x.shape  # Reads B and T used in Equation (A2).

        if h0 is None:
            # Equation (A2): use h_0 = 0 when no initial state is supplied.
            h_t = self.initial_hidden(batch_size, x.device, x.dtype)
        else:
            # Uses the provided initial state as h_0 in Equation (A2).
            h_t = h0

        outputs = []  # Will store h_1, ..., h_T from Equation (A2).

        for t in range(seq_len):
            # Reads x_t from the sequence x = [x_1, ..., x_T].
            x_t = x[:, t, :]
            # Applies Equation set (A1) to get the next hidden state h_t.
            h_t = self.cell(x_t, h_t)
            # Appends h_t so we can later build H = [h_1, ..., h_T].
            outputs.append(h_t)

        # Equation (A2): stack [h_1, ..., h_T] into H with shape [B, T, d_h].
        H = torch.stack(outputs, dim=1)
        # Returns the full hidden sequence H and the final hidden state h_T.
        return H, h_t
