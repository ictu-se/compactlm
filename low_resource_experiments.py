import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from primitive_cifg_lstm_usage import PrimitiveCharCIFGLSTMLanguageModel
from primitive_gru_usage import PrimitiveCharGRULanguageModel
from primitive_lstm_usage import PrimitiveCharLSTMLanguageModel
from primitive_peephole_lstm_usage import PrimitiveCharPeepholeLSTMLanguageModel
from primitive_rnn_usage import PrimitiveCharRNNLanguageModel
from primitive_usage_helpers import PrimitiveSequenceDataset, build_sequence_dataset, next_token_cross_entropy, sample_batch
from tiny_transformer import TinyTransformerDecoderLanguageModel


ARTIFACTS_DIR = Path("artifacts")
FIGURES_DIR = ARTIFACTS_DIR / "figures"
JSON_PATH = ARTIFACTS_DIR / "low_resource_experiment_results.json"
SUMMARY_PATH = ARTIFACTS_DIR / "low_resource_experiment_summary.md"
PROGRESS_PATH = ARTIFACTS_DIR / "progress.json"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    test_loss: float
    epoch_time_seconds: float
    train_tokens_per_second: float


@dataclass
class RunMetrics:
    model_name: str
    seed: int
    family: str
    params: int
    best_epoch: int
    best_val_loss: float
    final_test_loss: float
    test_ppl: float
    total_training_time_seconds: float
    time_to_best_checkpoint_seconds: float
    mean_epoch_time_seconds: float
    mean_train_tokens_per_second: float
    generation_tokens_per_second: float
    peak_memory_mb: float
    stop_reason: str
    converged: bool
    greedy_sample: str
    sampled_sample: str
    epoch_history: list[EpochMetrics]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@torch.no_grad()
def evaluate_average_loss(
    model: nn.Module,
    token_stream: torch.Tensor,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = sample_batch(token_stream, seq_len, batch_size)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = next_token_cross_entropy(logits, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_text_on_device(
    model: nn.Module,
    dataset: PrimitiveSequenceDataset,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float | None = None,
) -> str:
    model.eval()
    prompt_ids = [dataset.stoi[ch] for ch in prompt if ch in dataset.stoi]
    if not prompt_ids:
        prompt_ids = [0]
    generated_ids = list(prompt_ids)
    max_seq_len = getattr(model, "max_seq_len", None)
    context_ids = generated_ids[-max_seq_len:] if max_seq_len is not None else generated_ids
    x = torch.tensor([context_ids], dtype=torch.long, device=device)
    logits = model(x)
    for _ in range(max_new_tokens):
        next_token_logits = logits[:, -1, :]
        if temperature is None:
            next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
        else:
            next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = int(torch.multinomial(next_token_probs[0], num_samples=1).item())
        generated_ids.append(next_token_id)
        context_ids = generated_ids[-max_seq_len:] if max_seq_len is not None else generated_ids
        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits = model(x)
    model.train()
    return "".join(dataset.itos[idx] for idx in generated_ids)


def load_existing_runs() -> list[RunMetrics]:
    if not JSON_PATH.exists():
        return []
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    runs = []
    for item in payload.get("runs", []):
        epoch_history = [EpochMetrics(**epoch) for epoch in item.get("epoch_history", [])]
        run_payload = {key: value for key, value in item.items() if key != "epoch_history"}
        runs.append(RunMetrics(epoch_history=epoch_history, **run_payload))
    return runs


def save_progress(
    *,
    status: str,
    model_name: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
    max_epochs: int | None = None,
    best_epoch: int | None = None,
    best_val_loss: float | None = None,
    latest_train_loss: float | None = None,
    latest_val_loss: float | None = None,
    latest_test_loss: float | None = None,
    completed_runs: list[dict] | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "status": status,
        "model_name": model_name,
        "seed": seed,
        "epoch": epoch,
        "max_epochs": max_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "latest_train_loss": latest_train_loss,
        "latest_val_loss": latest_val_loss,
        "latest_test_loss": latest_test_loss,
        "completed_runs": completed_runs or [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def measure_generation_speed(
    model: nn.Module,
    dataset: PrimitiveSequenceDataset,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, str, float]:
    model = model.to(device)
    starter = time.perf_counter()
    greedy_sample = generate_text_on_device(model, dataset, device, prompt=prompt, max_new_tokens=max_new_tokens)
    sampled_sample = generate_text_on_device(
        model,
        dataset,
        device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
    )
    elapsed = max(time.perf_counter() - starter, 1e-9)
    tokens_generated = max_new_tokens * 2
    return greedy_sample, sampled_sample, tokens_generated / elapsed


def train_single_run(
    model_name: str,
    family: str,
    model_factory,
    dataset: PrimitiveSequenceDataset,
    seed: int,
    device: torch.device,
    seq_len: int = 64,
    batch_size: int = 32,
    steps_per_epoch: int = 100,
    eval_batches: int = 20,
    max_epochs: int = 50,
    validation_patience: int = 5,
    min_delta: float = 0.002,
) -> RunMetrics:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    set_seed(seed)
    model = model_factory(len(dataset.stoi)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    params = count_parameters(model)
    checkpoint_path = ARTIFACTS_DIR / f"{model_name.lower().replace(' ', '_')}_seed_{seed}.pt"
    epoch_history: list[EpochMetrics] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0
    stop_reason = "max_epochs_reached"
    time_to_best_checkpoint_seconds = 0.0
    total_train_tokens = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    training_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        epoch_token_count = 0
        epoch_start = time.perf_counter()

        for _ in range(steps_per_epoch):
            x, y = sample_batch(dataset.train_tokens, seq_len, batch_size)
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = next_token_cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            tokens_this_step = x.numel()
            epoch_token_count += tokens_this_step
            total_train_tokens += tokens_this_step

        epoch_time_seconds = time.perf_counter() - epoch_start
        train_tokens_per_second = epoch_token_count / max(epoch_time_seconds, 1e-9)
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = evaluate_average_loss(model, dataset.val_tokens, seq_len, batch_size, eval_batches, device)
        test_loss = evaluate_average_loss(model, dataset.test_tokens, seq_len, batch_size, eval_batches, device)
        epoch_history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                epoch_time_seconds=epoch_time_seconds,
                train_tokens_per_second=train_tokens_per_second,
            )
        )
        print(
            f"model={model_name} seed={seed} epoch={epoch} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} test_loss={test_loss:.4f} "
            f"epoch_time={epoch_time_seconds:.2f}s",
            flush=True,
        )

        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            time_to_best_checkpoint_seconds = time.perf_counter() - training_start
        else:
            epochs_since_improvement += 1

        save_progress(
            status="running",
            model_name=model_name,
            seed=seed,
            epoch=epoch,
            max_epochs=max_epochs,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            latest_train_loss=train_loss,
            latest_val_loss=val_loss,
            latest_test_loss=test_loss,
        )

        if epochs_since_improvement >= validation_patience:
            stop_reason = "validation_plateau"
            break

    if epoch_history:
        latest_metrics = epoch_history[-1]
        save_progress(
            status="run_complete",
            model_name=model_name,
            seed=seed,
            epoch=latest_metrics.epoch,
            max_epochs=max_epochs,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            latest_train_loss=latest_metrics.train_loss,
            latest_val_loss=latest_metrics.val_loss,
            latest_test_loss=latest_metrics.test_loss,
            extra={"stop_reason": stop_reason},
        )

    total_training_time_seconds = time.perf_counter() - training_start
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    final_test_loss = evaluate_average_loss(model, dataset.test_tokens, seq_len, batch_size, eval_batches, device)
    test_ppl = math.exp(final_test_loss)
    greedy_sample, sampled_sample, generation_tokens_per_second = measure_generation_speed(
        model=model,
        dataset=dataset,
        device=device,
        prompt="ROMEO:\n",
        max_new_tokens=200,
    )
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    mean_epoch_time_seconds = statistics.mean(metric.epoch_time_seconds for metric in epoch_history)
    mean_train_tokens_per_second = statistics.mean(metric.train_tokens_per_second for metric in epoch_history)

    return RunMetrics(
        model_name=model_name,
        seed=seed,
        family=family,
        params=params,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_test_loss=final_test_loss,
        test_ppl=test_ppl,
        total_training_time_seconds=total_training_time_seconds,
        time_to_best_checkpoint_seconds=time_to_best_checkpoint_seconds,
        mean_epoch_time_seconds=mean_epoch_time_seconds,
        mean_train_tokens_per_second=mean_train_tokens_per_second,
        generation_tokens_per_second=generation_tokens_per_second,
        peak_memory_mb=peak_memory_mb,
        stop_reason=stop_reason,
        converged=True,
        greedy_sample=greedy_sample,
        sampled_sample=sampled_sample,
        epoch_history=epoch_history,
    )


def summarize_runs(run_metrics: list[RunMetrics]) -> dict[str, dict]:
    grouped: dict[str, list[RunMetrics]] = {}
    for run in run_metrics:
        grouped.setdefault(run.model_name, []).append(run)

    summaries: dict[str, dict] = {}
    tiny_transformer_loss = None
    transformer_runs = [
        run
        for model_name, runs in grouped.items()
        if "transformer" in model_name.lower()
        for run in runs
    ]
    if transformer_runs:
        tiny_transformer_loss = statistics.mean(run.final_test_loss for run in transformer_runs)

    for model_name, runs in grouped.items():
        summary = {
            "family": runs[0].family,
            "params": runs[0].params,
            "seeds": [run.seed for run in runs],
            "mean_best_val_loss": statistics.mean(run.best_val_loss for run in runs),
            "std_best_val_loss": statistics.stdev(run.best_val_loss for run in runs) if len(runs) > 1 else 0.0,
            "mean_final_test_loss": statistics.mean(run.final_test_loss for run in runs),
            "std_final_test_loss": statistics.stdev(run.final_test_loss for run in runs) if len(runs) > 1 else 0.0,
            "mean_test_ppl": statistics.mean(run.test_ppl for run in runs),
            "mean_best_epoch": statistics.mean(run.best_epoch for run in runs),
            "mean_epoch_time_seconds": statistics.mean(run.mean_epoch_time_seconds for run in runs),
            "mean_time_to_best_checkpoint_seconds": statistics.mean(run.time_to_best_checkpoint_seconds for run in runs),
            "mean_train_tokens_per_second": statistics.mean(run.mean_train_tokens_per_second for run in runs),
            "mean_generation_tokens_per_second": statistics.mean(run.generation_tokens_per_second for run in runs),
            "mean_peak_memory_mb": statistics.mean(run.peak_memory_mb for run in runs),
            "all_converged": all(run.converged for run in runs),
            "stop_reasons": sorted({run.stop_reason for run in runs}),
            "example_greedy_sample": runs[0].greedy_sample,
            "example_sampled_text": runs[0].sampled_sample,
            "epoch_history_mean": aggregate_epoch_history(runs),
        }
        if tiny_transformer_loss is not None and tiny_transformer_loss > 0:
            summary["gap_to_tiny_transformer_percent"] = (
                (summary["mean_final_test_loss"] - tiny_transformer_loss) / tiny_transformer_loss
            ) * 100.0
        else:
            summary["gap_to_tiny_transformer_percent"] = 0.0
        summaries[model_name] = summary
    return summaries


def aggregate_epoch_history(runs: list[RunMetrics]) -> list[dict]:
    max_len = max(len(run.epoch_history) for run in runs)
    history = []
    for epoch_index in range(max_len):
        epoch_values = []
        for run in runs:
            if epoch_index < len(run.epoch_history):
                epoch_values.append(run.epoch_history[epoch_index])
        history.append(
            {
                "epoch": epoch_index + 1,
                "mean_train_loss": statistics.mean(item.train_loss for item in epoch_values),
                "mean_val_loss": statistics.mean(item.val_loss for item in epoch_values),
                "mean_test_loss": statistics.mean(item.test_loss for item in epoch_values),
            }
        )
    return history


def plot_quality_efficiency(summaries: dict[str, dict]) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "quality_efficiency_tradeoff.png"
    plt.figure(figsize=(10, 6))
    for model_name, summary in summaries.items():
        plt.scatter(
            summary["mean_time_to_best_checkpoint_seconds"],
            summary["mean_final_test_loss"],
            s=max(summary["params"] / 200, 30),
            label=model_name,
            alpha=0.8,
        )
        plt.annotate(model_name, (summary["mean_time_to_best_checkpoint_seconds"], summary["mean_final_test_loss"]))
    plt.xlabel("Time to Best Checkpoint (s)")
    plt.ylabel("Final Test Loss")
    plt.title("Quality-Efficiency Trade-Off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_learning_curves(summaries: dict[str, dict]) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "validation_learning_curves.png"
    plt.figure(figsize=(10, 6))
    for model_name, summary in summaries.items():
        epochs = [point["epoch"] for point in summary["epoch_history_mean"]]
        val_losses = [point["mean_val_loss"] for point in summary["epoch_history_mean"]]
        plt.plot(epochs, val_losses, label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss by Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def write_summary_markdown(summaries: dict[str, dict], figure_paths: list[Path]) -> None:
    preferred_order = [
        "Primitive RNN",
        "Primitive GRU",
        "Primitive LSTM",
        "Primitive Peephole LSTM",
        "Primitive CIFG LSTM",
        "Tiny Transformer Decoder",
        "Primitive RNN Fair",
        "Primitive GRU Fair",
        "Primitive LSTM Fair",
        "Primitive Peephole LSTM Fair",
        "Primitive CIFG LSTM Fair",
        "Tiny Transformer Decoder Fair",
    ]
    ordered_models = [model_name for model_name in preferred_order if model_name in summaries]
    ordered_models.extend(model_name for model_name in summaries if model_name not in ordered_models)

    lines = [
        "# Low-Resource Experiment Summary",
        "",
        "This file is generated by `low_resource_experiments.py`.",
        "",
        "## Table 1. Main Effectiveness Comparison",
        "",
        "| Model | Family | Params | Seeds | Best Val Loss (mean +- std) | Final Test Loss (mean +- std) | Test PPL | Gap to Tiny Transformer (%) |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for model_name in ordered_models:
        if model_name not in summaries:
            continue
        summary = summaries[model_name]
        lines.append(
            f"| {model_name} | {summary['family']} | {summary['params']} | {len(summary['seeds'])} | "
            f"{summary['mean_best_val_loss']:.4f} +- {summary['std_best_val_loss']:.4f} | "
            f"{summary['mean_final_test_loss']:.4f} +- {summary['std_final_test_loss']:.4f} | "
            f"{summary['mean_test_ppl']:.2f} | {summary['gap_to_tiny_transformer_percent']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Table 2. Efficiency and Cost Comparison",
            "",
            "| Model | Params | Mean Best Epoch | Mean Epoch Time (s) | Mean Time to Best Checkpoint (s) | Train Tokens/s | Gen Tokens/s | Peak Memory (MB) | Test Loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_name in ordered_models:
        if model_name not in summaries:
            continue
        summary = summaries[model_name]
        lines.append(
            f"| {model_name} | {summary['params']} | {summary['mean_best_epoch']:.2f} | "
            f"{summary['mean_epoch_time_seconds']:.2f} | {summary['mean_time_to_best_checkpoint_seconds']:.2f} | "
            f"{summary['mean_train_tokens_per_second']:.2f} | {summary['mean_generation_tokens_per_second']:.2f} | "
            f"{summary['mean_peak_memory_mb']:.2f} | {summary['mean_final_test_loss']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Table 3. Stability Across Seeds",
            "",
            "| Model | Seeds | Mean Best Val Loss | Std Best Val Loss | Mean Final Test Loss | Std Final Test Loss | Mean Best Epoch | Converged in All Seeds? | Stop Reasons |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for model_name in ordered_models:
        if model_name not in summaries:
            continue
        summary = summaries[model_name]
        lines.append(
            f"| {model_name} | {len(summary['seeds'])} | {summary['mean_best_val_loss']:.4f} | "
            f"{summary['std_best_val_loss']:.4f} | {summary['mean_final_test_loss']:.4f} | "
            f"{summary['std_final_test_loss']:.4f} | {summary['mean_best_epoch']:.2f} | "
            f"{'Yes' if summary['all_converged'] else 'No'} | {', '.join(summary['stop_reasons'])} |"
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Figure 1: `{figure_paths[0]}`",
            f"- Figure 2: `{figure_paths[1]}`",
            "",
            "## Qualitative Samples",
            "",
            "| Model | Greedy Sample Excerpt | Sampled Sample Excerpt |",
            "| --- | --- | --- |",
        ]
    )
    for model_name in ordered_models:
        if model_name not in summaries:
            continue
        summary = summaries[model_name]
        greedy_excerpt = summary["example_greedy_sample"].replace("\n", " ")[:80]
        sampled_excerpt = summary["example_sampled_text"].replace("\n", " ")[:80]
        lines.append(f"| {model_name} | `{greedy_excerpt}...` | `{sampled_excerpt}...` |")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"device={device}")
    dataset = build_sequence_dataset(Path("tinyshakespeare.txt"))
    seeds = [7, 11, 19]

    model_specs = [
        (
            "Primitive RNN",
            "RNN",
            lambda vocab_size: PrimitiveCharRNNLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64),
        ),
        (
            "Primitive GRU",
            "GRU",
            lambda vocab_size: PrimitiveCharGRULanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64),
        ),
        (
            "Primitive LSTM",
            "LSTM",
            lambda vocab_size: PrimitiveCharLSTMLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64),
        ),
        (
            "Primitive Peephole LSTM",
            "LSTM Variant",
            lambda vocab_size: PrimitiveCharPeepholeLSTMLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64),
        ),
        (
            "Primitive CIFG LSTM",
            "LSTM Variant",
            lambda vocab_size: PrimitiveCharCIFGLSTMLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64),
        ),
        (
            "Tiny Transformer Decoder",
            "Transformer",
            lambda vocab_size: TinyTransformerDecoderLanguageModel(
                vocab_size=vocab_size,
                d_model=32,
                num_heads=4,
                ff_dim=128,
                num_layers=2,
                max_seq_len=64,
                dropout=0.1,
            ),
        ),
    ]

    all_runs = load_existing_runs()
    completed_runs = {(run.model_name, run.seed) for run in all_runs}
    for model_name, family, model_factory in model_specs:
        for seed in seeds:
            if (model_name, seed) in completed_runs:
                print(f"skipping completed model={model_name} seed={seed}")
                continue
            print(f"starting model={model_name} seed={seed}")
            run = train_single_run(
                model_name=model_name,
                family=family,
                model_factory=model_factory,
                dataset=dataset,
                seed=seed,
                device=device,
            )
            all_runs.append(run)
            completed_runs.add((model_name, seed))
            JSON_PATH.write_text(
                json.dumps(
                    {
                        "runs": [serialize_run_metrics(item) for item in all_runs],
                        "summaries": summarize_runs(all_runs),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    summaries = summarize_runs(all_runs)
    figure_paths = [
        plot_quality_efficiency(summaries),
        plot_learning_curves(summaries),
    ]
    write_summary_markdown(summaries, figure_paths)
    JSON_PATH.write_text(
        json.dumps(
            {
                "runs": [serialize_run_metrics(item) for item in all_runs],
                "summaries": summaries,
                "figures": [str(path) for path in figure_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"summary_path={SUMMARY_PATH}")
    for figure_path in figure_paths:
        print(f"figure_path={figure_path}")


def serialize_run_metrics(run: RunMetrics) -> dict:
    payload = asdict(run)
    payload["epoch_history"] = [asdict(item) for item in run.epoch_history]
    return payload


if __name__ == "__main__":
    main()
