import torch
from torch import nn

# ARCHITECTURE FILE
# This file contains only the primitive LSTM equations and their direct implementation.
# It does not contain data loading, optimization, early stopping, or text generation policy.


class PrimitiveLSTMCell(nn.Module):
    # Equation set (A1) in primitive_lstm.md:
    #   f_t = sigmoid(x_t W_xf + h_{t-1} W_hf + b_f)
    #   i_t = sigmoid(x_t W_xi + h_{t-1} W_hi + b_i)
    #   o_t = sigmoid(x_t W_xo + h_{t-1} W_ho + b_o)
    #   g_t = tanh(x_t W_xg + h_{t-1} W_hg + b_g)
    #   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    #   h_t = o_t ⊙ tanh(c_t)
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()  # Standard nn.Module initialization; not part of the math itself.
        self.input_dim = input_dim  # Stores d_x, the width of x_t in Equation set (A1).
        self.hidden_dim = hidden_dim  # Stores d_h, the width of h_t and c_t in Equation set (A1).

        # Equation (A1): W_xf, W_hf, b_f.
        self.W_xf = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_hf = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))

        # Equation (A1): W_xi, W_hi, b_i.
        self.W_xi = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_hi = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))

        # Equation (A1): W_xo, W_ho, b_o.
        self.W_xo = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_ho = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

        # Equation (A1): W_xg, W_hg, b_g.
        self.W_xg = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W_hg = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))

        self.reset_parameters()  # Chooses initial numeric values for the symbols in Equation set (A1).

    def reset_parameters(self) -> None:
        # This is initialization only; it is not one of the forward equations in the markdown.
        nn.init.xavier_uniform_(self.W_xf)
        nn.init.orthogonal_(self.W_hf)
        nn.init.zeros_(self.b_f)
        nn.init.xavier_uniform_(self.W_xi)
        nn.init.orthogonal_(self.W_hi)
        nn.init.zeros_(self.b_i)
        nn.init.xavier_uniform_(self.W_xo)
        nn.init.orthogonal_(self.W_ho)
        nn.init.zeros_(self.b_o)
        nn.init.xavier_uniform_(self.W_xg)
        nn.init.orthogonal_(self.W_hg)
        nn.init.zeros_(self.b_g)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Equation (A1): f_t = sigmoid(x_t W_xf + h_{t-1} W_hf + b_f).
        f_t = torch.sigmoid((x_t @ self.W_xf) + (h_prev @ self.W_hf) + self.b_f)
        # Equation (A1): i_t = sigmoid(x_t W_xi + h_{t-1} W_hi + b_i).
        i_t = torch.sigmoid((x_t @ self.W_xi) + (h_prev @ self.W_hi) + self.b_i)
        # Equation (A1): o_t = sigmoid(x_t W_xo + h_{t-1} W_ho + b_o).
        o_t = torch.sigmoid((x_t @ self.W_xo) + (h_prev @ self.W_ho) + self.b_o)
        # Equation (A1): g_t = tanh(x_t W_xg + h_{t-1} W_hg + b_g).
        g_t = torch.tanh((x_t @ self.W_xg) + (h_prev @ self.W_hg) + self.b_g)
        # Equation (A1): c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t.
        c_t = (f_t * c_prev) + (i_t * g_t)
        # Equation (A1): h_t = o_t ⊙ tanh(c_t).
        h_t = o_t * torch.tanh(c_t)
        # Returns the pair (h_t, c_t) from Equation set (A1).
        return h_t, c_t


class PrimitiveLSTMBackbone(nn.Module):
    # Equation set (A2) in primitive_lstm.md:
    #   h_0 = 0
    #   c_0 = 0
    #   (h_t, c_t) = LSTMCell(x_t, h_{t-1}, c_{t-1}),  t = 1, ..., T
    #   H = [h_1, h_2, ..., h_T]
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()  # Standard nn.Module initialization.
        self.input_dim = input_dim  # Stores d_x from Equation (A2).
        self.hidden_dim = hidden_dim  # Stores d_h from Equation (A2).
        self.cell = PrimitiveLSTMCell(input_dim, hidden_dim)  # Uses Equation set (A1) at each time step.

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Equation (A2): h_0 = 0 and c_0 = 0.
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        return h0, c0

    def forward(
        self,
        x: torch.Tensor,
        state0: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x has shape [B, T, d_x], which is the batch of sequences x_1, ..., x_T from Equation (A2).
        batch_size, seq_len, _ = x.shape  # Reads B and T used in Equation (A2).

        if state0 is None:
            # Equation (A2): use h_0 = 0 and c_0 = 0 when no initial state is supplied.
            h_t, c_t = self.initial_state(batch_size, x.device, x.dtype)
        else:
            # Uses the provided initial state as (h_0, c_0) in Equation (A2).
            h_t, c_t = state0

        outputs = []  # Will store h_1, ..., h_T from Equation (A2).

        for t in range(seq_len):
            # Reads x_t from the sequence x = [x_1, ..., x_T].
            x_t = x[:, t, :]
            # Applies Equation set (A1) to get the next pair (h_t, c_t).
            h_t, c_t = self.cell(x_t, h_t, c_t)
            # Appends h_t so we can later build H = [h_1, ..., h_T].
            outputs.append(h_t)

        # Equation (A2): stack [h_1, ..., h_T] into H with shape [B, T, d_h].
        H = torch.stack(outputs, dim=1)
        # Returns the full hidden sequence H and the final pair (h_T, c_T).
        return H, (h_t, c_t)
