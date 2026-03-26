from torch import nn

from primitive_cifg_lstm import PrimitiveCIFGLSTMBackbone
from primitive_usage_helpers import train_variant_until_test_reverses

# USAGE FILE
# This file does not define the primitive CIFG LSTM architecture itself.
# It only shows how to use the architecture inside a character-level language-model workflow.


class PrimitiveCharCIFGLSTMLanguageModel(nn.Module):
    # Equation set (B1) in primitive_cifg_lstm_usage.md:
    #   e_t = Embedding(x_t)
    #   h_t = PrimitiveCIFGLSTM(e_t, h_{t-1}, c_{t-1})
    #   z_t = h_t W_hy + b_y
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.backbone = PrimitiveCIFGLSTMBackbone(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        hidden_sequence, _ = self.backbone(embedded)
        logits = self.output(hidden_sequence)
        return logits


def train_until_test_reverses() -> None:
    train_variant_until_test_reverses(
        model_factory=lambda vocab_size: PrimitiveCharCIFGLSTMLanguageModel(vocab_size, embed_dim=32, hidden_dim=64),
        checkpoint_filename="primitive_cifg_lstm_best_checkpoint.pt",
        report_filename="primitive_cifg_lstm_report.md",
        report_title="Primitive CIFG LSTM Report",
        usage_filename="primitive_cifg_lstm_usage.py",
        architecture_filename="primitive_cifg_lstm.py",
        usage_markdown_filename="primitive_cifg_lstm_usage.md",
        explanation_heading="Why CIFG LSTM Behaves Differently",
        explanation_lines=[
            "This variant couples the input and forget decisions through i_t = 1 - f_t.",
            "That reduces the number of gates and slightly simplifies the memory update.",
            "It can be seen as a lighter LSTM that still keeps an explicit cell state.",
            "So it is useful when you want to compare memory behavior with fewer gate degrees of freedom.",
        ],
    )


if __name__ == "__main__":
    train_until_test_reverses()
