from torch import nn

from primitive_peephole_lstm import PrimitivePeepholeLSTMBackbone
from primitive_usage_helpers import train_variant_until_test_reverses

# USAGE FILE
# This file does not define the primitive peephole LSTM architecture itself.
# It only shows how to use the architecture inside a character-level language-model workflow.


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


def train_until_test_reverses() -> None:
    train_variant_until_test_reverses(
        model_factory=lambda vocab_size: PrimitiveCharPeepholeLSTMLanguageModel(vocab_size, embed_dim=32, hidden_dim=64),
        checkpoint_filename="primitive_peephole_lstm_best_checkpoint.pt",
        report_filename="primitive_peephole_lstm_report.md",
        report_title="Primitive Peephole LSTM Report",
        usage_filename="primitive_peephole_lstm_usage.py",
        architecture_filename="primitive_peephole_lstm.py",
        usage_markdown_filename="primitive_peephole_lstm_usage.md",
        explanation_heading="Why Peephole LSTM Behaves Differently",
        explanation_lines=[
            "The gates can inspect the cell state directly through the peephole terms.",
            "That can help the model make finer gating decisions about memory timing.",
            "It is still a causal recurrent model, unlike the bidirectional variants.",
            "So it is a closer drop-in comparison against the ordinary LSTM.",
        ],
    )


if __name__ == "__main__":
    train_until_test_reverses()
