from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from primitive_rnn import PrimitiveRNNBackbone

# USAGE FILE
# This file does not define the primitive RNN architecture itself.
# It only shows how to use the architecture inside a character-level language-model workflow:
# dataset split, batching, loss computation, optimization, validation, early stopping, and sampling.


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


class PrimitiveCharRNNLanguageModel(nn.Module):
    # Equation set (B1) in primitive_rnn_usage.md:
    #   e_t = Embedding(x_t)
    #   h_t = PrimitiveRNN(e_t, h_{t-1})
    #   z_t = h_t W_hy + b_y
    #   p(x_{t+1} | x_{<=t}) = softmax(z_t)
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()  # Standard nn.Module initialization.
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Equation (B1): e_t = Embedding(x_t).
        self.backbone = PrimitiveRNNBackbone(embed_dim, hidden_dim)  # Equation (B1): h_t = PrimitiveRNN(e_t, h_{t-1}).
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


def sample_batch(token_stream: torch.Tensor, seq_len: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Randomly samples starting positions i for subsequences:
    #   x^(b) = [x_i, ..., x_{i+T-1}]
    #   y^(b) = [x_{i+1}, ..., x_{i+T}]
    starts = torch.randint(0, len(token_stream) - seq_len - 1, (batch_size,))
    # Collects the input subsequences x_1, ..., x_T for each batch element.
    x = torch.stack([token_stream[i : i + seq_len] for i in starts])
    # Collects the next-token targets shifted by one position.
    y = torch.stack([token_stream[i + 1 : i + seq_len + 1] for i in starts])
    # Returns the supervised pair (inputs, targets).
    return x, y


def next_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Equation (B2):
    #   L = CrossEntropy(z_t, target_t)
    # applied over every batch element and every time step.
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def generate_sample_text(
    model: PrimitiveCharRNNLanguageModel,
    dataset: PrimitiveSequenceDataset,
    prompt: str,
    max_new_tokens: int,
) -> str:
    # Switches to evaluation mode before autoregressive generation.
    model.eval()

    # Converts the text prompt into token ids x_1, ..., x_T.
    prompt_ids = [dataset.stoi[ch] for ch in prompt if ch in dataset.stoi]
    if not prompt_ids:
        prompt_ids = [0]

    # Stores the generated token ids; it starts with the prompt itself.
    generated_ids = list(prompt_ids)

    # Feeds the prompt through the model to obtain the hidden sequence and final logits.
    x = torch.tensor([prompt_ids], dtype=torch.long)
    logits = model(x)

    for _ in range(max_new_tokens):
        # Uses the last-step logits z_T to define p(x_{T+1} | x_{<=T}).
        next_token_logits = logits[:, -1, :]
        # Chooses the most likely next token id.
        next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
        # Appends the chosen token to the generated sequence.
        generated_ids.append(next_token_id)
        # Re-runs the model on the full generated prefix x_1, ..., x_{T+1}.
        x = torch.tensor([generated_ids], dtype=torch.long)
        logits = model(x)

    # Switches back to training mode after generation.
    model.train()
    # Decodes generated ids back into characters.
    return "".join(dataset.itos[idx] for idx in generated_ids)


@torch.no_grad()
def generate_sample_text_with_sampling(
    model: PrimitiveCharRNNLanguageModel,
    dataset: PrimitiveSequenceDataset,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    # Switches to evaluation mode before probabilistic autoregressive generation.
    model.eval()

    # Converts the text prompt into token ids x_1, ..., x_T.
    prompt_ids = [dataset.stoi[ch] for ch in prompt if ch in dataset.stoi]
    if not prompt_ids:
        prompt_ids = [0]

    # Stores the generated token ids; it starts with the prompt itself.
    generated_ids = list(prompt_ids)

    # Feeds the prompt through the model to obtain the hidden sequence and final logits.
    x = torch.tensor([prompt_ids], dtype=torch.long)
    logits = model(x)

    for _ in range(max_new_tokens):
        # Uses a temperature-scaled distribution:
        #   p(x_{t+1} | x_{<=t}) = softmax(z_t / tau)
        next_token_logits = logits[:, -1, :] / temperature
        # Converts logits into probabilities over the vocabulary.
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        # Samples one token id from the categorical distribution instead of taking argmax.
        next_token_id = int(torch.multinomial(next_token_probs[0], num_samples=1).item())
        # Appends the sampled token to the generated sequence.
        generated_ids.append(next_token_id)
        # Re-runs the model on the full generated prefix x_1, ..., x_{T+1}.
        x = torch.tensor([generated_ids], dtype=torch.long)
        logits = model(x)

    # Switches back to training mode after generation.
    model.train()
    # Decodes generated ids back into characters.
    return "".join(dataset.itos[idx] for idx in generated_ids)


@torch.no_grad()
def evaluate_average_loss(
    model: PrimitiveCharRNNLanguageModel,
    token_stream: torch.Tensor,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
) -> float:
    # Switches the model to evaluation mode for validation measurement.
    model.eval()
    losses = []
    for _ in range(eval_batches):
        # Draws one validation mini-batch from D_val.
        x, y = sample_batch(token_stream, seq_len, batch_size)
        # Computes logits z_t under Equation (B1).
        logits = model(x)
        # Computes the batch loss under Equation (B2).
        loss = next_token_cross_entropy(logits, y)
        losses.append(loss.item())
    # Restores training mode after validation.
    model.train()
    # Returns the empirical mean validation loss.
    return sum(losses) / len(losses)


def write_primitive_rnn_report(
    report_path: Path,
    epoch_reports: list[PrimitiveEpochReport],
    best_epoch: int,
    best_test_loss: float,
    checkpoint_path: Path,
    stop_reason: str,
    reversal_threshold: float,
    greedy_sample_text: str,
    sampled_sample_text: str,
    sampling_temperature: float,
) -> None:
    # Builds a Markdown report so the architecture has a readable experiment summary.
    lines = [
        "# Primitive RNN Report",
        "",
        "This report is produced by `primitive_rnn_usage.py` for the architecture in `primitive_rnn.py`.",
        "",
        "## Run Summary",
        "",
        f"- Best epoch: `{best_epoch}`",
        f"- Best test loss: `{best_test_loss:.4f}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Stop reason: `{stop_reason}`",
        f"- Test-reversal threshold: `{reversal_threshold:.4f}`",
        "",
        "## Epoch History",
        "",
        "| Epoch | Train Loss | Val Loss | Test Loss |",
        "| --- | ---: | ---: | ---: |",
    ]

    for epoch_report in epoch_reports:
        # Adds one row R_k = (k, L_train^(k), L_val^(k), L_test^(k)) to the report table.
        lines.append(
            f"| {epoch_report.epoch} | "
            f"{epoch_report.train_loss:.4f} | "
            f"{epoch_report.val_loss:.4f} | "
            f"{epoch_report.test_loss:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Sample Generation",
            "",
            "The samples below are generated from the best checkpoint using two decoding styles.",
            "",
            "### Greedy Decoding",
            "",
            "This version always takes the most likely next token.",
            "",
            "```text",
            greedy_sample_text,
            "```",
            "",
            "### Probabilistic Sampling",
            "",
            f"This version samples from `softmax(z_t / tau)` with `tau = {sampling_temperature:.2f}`.",
            "",
            "```text",
            sampled_sample_text,
            "```",
            "",
            "## Interpretation",
            "",
            "- This report belongs to the primitive RNN architecture only.",
            "- The architecture equations are documented in `primitive_rnn.md`.",
            "- The training workflow equations are documented in `primitive_rnn_usage.md`.",
            "- Greedy decoding exposes repetition very clearly because the model keeps selecting the local maximum-probability token.",
            "- Probabilistic sampling can reduce hard repetition, but weak models often become noisier instead.",
            "- A repetitive sample is normal here because primitive RNNs struggle with long context.",
        ]
    )

    # Writes the Markdown report to disk.
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_until_test_reverses() -> None:
    # Loads the dataset used for the primitive-RNN training demonstration.
    dataset = build_sequence_dataset(Path("tinyshakespeare.txt"))
    # Instantiates the language model from Equation set (B1).
    model = PrimitiveCharRNNLanguageModel(vocab_size=len(dataset.stoi), embed_dim=32, hidden_dim=64)
    # Uses Adam to optimize the parameters appearing in Equations (A1), (A2), and (B1).
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Chooses the fixed training hyperparameters for this educational run.
    seq_len = 64
    batch_size = 32
    steps_per_epoch = 100
    eval_batches = 20
    max_epochs = 50

    previous_test_loss = None  # Stores L_test^(k-1) so we can detect when L_test^(k) increases.
    best_test_loss = None  # Stores the best test loss seen so far.
    best_epoch = None  # Stores the epoch index that achieved the best test loss.
    checkpoint_path = Path("primitive_rnn_best_checkpoint.pt")  # Stores theta* from the best test epoch.
    report_path = Path("primitive_rnn_report.md")  # Stores a human-readable report for this architecture run.
    reversal_threshold = 0.01  # Requires a noticeable upward move in test loss before stopping.
    epoch_reports = []  # Stores the sequence R_1, R_2, ..., R_k used in the report table.
    stop_reason = "max_epochs_reached"  # Default stop reason unless the test loss reverses first.

    for epoch in range(1, max_epochs + 1):
        train_losses = []

        for _ in range(steps_per_epoch):
            # Samples one training mini-batch from D_train.
            x, y = sample_batch(dataset.train_tokens, seq_len, batch_size)
            # Computes logits z_t from Equation (B1).
            logits = model(x)
            # Computes the training objective L from Equation (B2).
            loss = next_token_cross_entropy(logits, y)

            # Resets accumulated gradients before the new optimization step.
            optimizer.zero_grad(set_to_none=True)
            # Computes gradients dL/dtheta for all parameters theta in the model.
            loss.backward()
            # Applies one optimizer step: theta <- theta - eta * grad(theta) approximately.
            optimizer.step()

            # Stores the batch loss so we can report the epoch-average training loss.
            train_losses.append(loss.item())

        # Computes the average training loss over the epoch.
        train_loss = sum(train_losses) / len(train_losses)
        # Computes the average validation loss on D_val.
        val_loss = evaluate_average_loss(model, dataset.val_tokens, seq_len, batch_size, eval_batches)
        # Computes the average test loss on D_test.
        test_loss = evaluate_average_loss(model, dataset.test_tokens, seq_len, batch_size, eval_batches)

        # Reports the triple (L_train^(k), L_val^(k), L_test^(k)) for the current epoch k.
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} test_loss={test_loss:.4f}")
        # Stores the epoch-level metrics R_k for the report.
        epoch_reports.append(PrimitiveEpochReport(epoch, train_loss, val_loss, test_loss))

        # Keeps the best parameters theta* such that L_test is minimal so far.
        if best_test_loss is None or test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        # Early-stop condition:
        #   stop if L_test^(k) > L_test^(k-1) + delta
        # which means test loss has reversed upward by a noticeable amount.
        if previous_test_loss is not None and test_loss > previous_test_loss + reversal_threshold:
            stop_reason = "test_reversed"
            print(
                "stop_reason=test_reversed "
                f"previous_test_loss={previous_test_loss:.4f} "
                f"current_test_loss={test_loss:.4f} "
                f"threshold={reversal_threshold:.4f}"
            )
            break

        # Updates L_test^(k-1) <- L_test^(k) for the next epoch comparison.
        previous_test_loss = test_loss

    # Restores theta* from the epoch with the best test loss.
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    # Reports the best checkpoint selected during training.
    print(f"best_epoch={best_epoch} best_test_loss={best_test_loss:.4f} checkpoint={checkpoint_path}")

    # Generates a greedy sample x_{T+1}, x_{T+2}, ... from the best checkpoint.
    greedy_sample = generate_sample_text(model, dataset, prompt="ROMEO:\n", max_new_tokens=200)
    sampling_temperature = 0.7
    sampled_sample = generate_sample_text_with_sampling(
        model,
        dataset,
        prompt="ROMEO:\n",
        max_new_tokens=200,
        temperature=sampling_temperature,
    )
    print("greedy_sample_start")
    print(greedy_sample)
    print("greedy_sample_end")
    print("sampled_sample_start")
    print(sampled_sample)
    print("sampled_sample_end")

    # Writes a report file so this architecture has one consolidated experiment summary.
    write_primitive_rnn_report(
        report_path=report_path,
        epoch_reports=epoch_reports,
        best_epoch=best_epoch,
        best_test_loss=best_test_loss,
        checkpoint_path=checkpoint_path,
        stop_reason=stop_reason,
        reversal_threshold=reversal_threshold,
        greedy_sample_text=greedy_sample,
        sampled_sample_text=sampled_sample,
        sampling_temperature=sampling_temperature,
    )
    print(f"report_path={report_path}")


if __name__ == "__main__":
    # Runs training until the test loss stops improving and turns upward clearly.
    train_until_test_reverses()
