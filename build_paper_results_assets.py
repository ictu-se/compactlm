from __future__ import annotations

import json
import math
import statistics
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


BASE_DIR = Path("artifacts") / "fair_matched_budget_30k_multidataset"
GEN_PATH = BASE_DIR / "generation_benchmark" / "results.json"
OUT_DIR = Path("artifacts") / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


DATASET_ORDER = [
    "tinyshakespeare",
    "alice_in_wonderland",
    "pride_and_prejudice",
    "sherlock_holmes",
]

DATASET_LABELS = {
    "tinyshakespeare": "Tiny Shakespeare",
    "alice_in_wonderland": "Alice",
    "pride_and_prejudice": "Pride and Prejudice",
    "sherlock_holmes": "Sherlock Holmes",
}

MODEL_ORDER = [
    "Primitive RNN Fair",
    "Primitive GRU Fair",
    "Primitive LSTM Fair",
    "Primitive Peephole LSTM Fair",
    "Primitive CIFG LSTM Fair",
    "Tiny Transformer Decoder Fair",
]

MODEL_SHORT = {
    "Primitive RNN Fair": "RNN",
    "Primitive GRU Fair": "GRU",
    "Primitive LSTM Fair": "LSTM",
    "Primitive Peephole LSTM Fair": "Peephole",
    "Primitive CIFG LSTM Fair": "CIFG",
    "Tiny Transformer Decoder Fair": "Transformer",
}


def load_dataset_summaries() -> dict[str, dict]:
    payload = {}
    for dataset_id in DATASET_ORDER:
        path = BASE_DIR / dataset_id / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        payload[dataset_id] = data["summaries"]
    return payload


def load_generation_results() -> dict:
    return json.loads(GEN_PATH.read_text(encoding="utf-8"))


def average_summary_metrics(dataset_summaries: dict[str, dict]) -> dict[str, dict]:
    avg: dict[str, dict] = {}
    for model_name in MODEL_ORDER:
        rows = [dataset_summaries[dataset_id][model_name] for dataset_id in DATASET_ORDER]
        avg[model_name] = {
            "mean_final_test_loss": statistics.mean(row["mean_final_test_loss"] for row in rows),
            "mean_time_to_best_checkpoint_seconds": statistics.mean(
                row["mean_time_to_best_checkpoint_seconds"] for row in rows
            ),
            "mean_generation_tokens_per_second": statistics.mean(row["mean_generation_tokens_per_second"] for row in rows),
            "mean_peak_memory_mb": statistics.mean(row["mean_peak_memory_mb"] for row in rows),
            "mean_std_final_test_loss": statistics.mean(row["std_final_test_loss"] for row in rows),
        }
    return avg


def average_generation_metrics(generation_payload: dict) -> dict[str, dict]:
    agg = {model_name: {"greedy_f4": [], "greedy_rep": [], "sampled_f4": [], "sampled_rep": []} for model_name in MODEL_ORDER}
    for dataset in generation_payload["datasets"]:
        for run in dataset["runs"]:
            agg[run["model_name"]]["greedy_f4"].append(run["greedy_metrics"]["char_fourgram_f1"])
            agg[run["model_name"]]["greedy_rep"].append(run["greedy_metrics"]["repetition_rate_4gram"])
            agg[run["model_name"]]["sampled_f4"].append(run["sampled_metrics"]["char_fourgram_f1"])
            agg[run["model_name"]]["sampled_rep"].append(run["sampled_metrics"]["repetition_rate_4gram"])
    return {
        model_name: {metric: statistics.mean(values) for metric, values in metrics.items()}
        for model_name, metrics in agg.items()
    }


def build_cross_dataset_loss_figure(dataset_summaries: dict[str, dict]) -> Path:
    matrix = np.array(
        [
            [dataset_summaries[dataset_id][model_name]["mean_final_test_loss"] for dataset_id in DATASET_ORDER]
            for model_name in MODEL_ORDER
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(matrix, cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(range(len(DATASET_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[item] for item in DATASET_ORDER], rotation=15, ha="right")
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_SHORT[item] for item in MODEL_ORDER])
    ax.set_title("Figure 1. Mean Test Loss Across Datasets")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Test Loss")
    fig.tight_layout()
    out_path = OUT_DIR / "figure1_cross_dataset_test_loss.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_generation_tradeoff_figure(generation_avg: dict[str, dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = {
        "Primitive RNN Fair": "#1f77b4",
        "Primitive GRU Fair": "#2ca02c",
        "Primitive LSTM Fair": "#ff7f0e",
        "Primitive Peephole LSTM Fair": "#8c564b",
        "Primitive CIFG LSTM Fair": "#17becf",
        "Tiny Transformer Decoder Fair": "#d62728",
    }
    for model_name in MODEL_ORDER:
        row = generation_avg[model_name]
        ax.scatter(
            row["greedy_rep"],
            row["greedy_f4"],
            s=160,
            color=colors[model_name],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.6,
        )
        ax.annotate(MODEL_SHORT[model_name], (row["greedy_rep"], row["greedy_f4"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Average Greedy 4-gram Repetition Rate")
    ax.set_ylabel("Average Greedy Character 4-gram F1")
    ax.set_title("Figure 2. Free-Running Generation Quality vs Repetition")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = OUT_DIR / "figure2_generation_tradeoff.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def wrap_block(text: str, width: int = 58, max_lines: int = 5) -> str:
    text = text.replace("\n", " ").strip()
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, width - 3)] + "..."
    return "\n".join(lines)


def build_qualitative_examples_figure(generation_payload: dict) -> Path:
    datasets = {item["dataset_id"]: item for item in generation_payload["datasets"]}
    selected_ids = DATASET_ORDER
    fig, axes = plt.subplots(4, 4, figsize=(20, 14))
    fig.suptitle("Figure 3. Qualitative Generation Examples", fontsize=14, y=0.985)

    for row_idx, dataset_id in enumerate(selected_ids):
        dataset = datasets[dataset_id]
        runs = dataset["runs"]
        best_loss = min(
            runs,
            key=lambda item: item["final_test_loss"],
        )
        best_recurrent = max(
            [r for r in runs if r["model_name"] != "Tiny Transformer Decoder Fair"],
            key=lambda item: item["greedy_metrics"]["char_fourgram_f1"],
        )
        best_sampled = max(
            [r for r in runs if r["model_name"] != "Tiny Transformer Decoder Fair"],
            key=lambda item: item["sampled_metrics"]["char_fourgram_f1"],
        )
        transformer = next(r for r in runs if r["model_name"] == "Tiny Transformer Decoder Fair")
        panel_runs = [best_loss, best_recurrent, best_sampled, transformer]
        panel_labels = [
            "Best Test-Loss Model",
            "Best Greedy Recurrent",
            "Best Sampled Recurrent",
            "Tiny Transformer",
        ]
        for col_idx, (run, panel_label) in enumerate(zip(panel_runs, panel_labels)):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            ex = run["examples"][0]
            title = f"{DATASET_LABELS[dataset_id]}: {panel_label}\n{MODEL_SHORT[run['model_name']]}"
            body = (
                f"Prompt\n{wrap_block(ex['prompt'], width=58, max_lines=4)}\n\n"
                f"Reference\n{wrap_block(ex['reference'], width=58, max_lines=4)}\n\n"
                f"Greedy continuation\n{wrap_block(ex['greedy_continuation'], width=58, max_lines=4)}"
            )
            ax.add_patch(
                Rectangle(
                    (0.012, 0.02),
                    0.976,
                    0.93,
                    transform=ax.transAxes,
                    facecolor="#f7f7f7",
                    edgecolor="#cccccc",
                    linewidth=0.8,
                )
            )
            ax.text(
                0.02,
                0.94,
                title,
                fontsize=10.2,
                fontweight="bold",
                va="top",
                ha="left",
                transform=ax.transAxes,
            )
            ax.text(
                0.02,
                0.825,
                body,
                fontsize=8.35,
                family="monospace",
                va="top",
                ha="left",
                linespacing=1.08,
                transform=ax.transAxes,
            )
    fig.subplots_adjust(left=0.01, right=0.992, top=0.94, bottom=0.02, wspace=0.006, hspace=0.10)
    out_path = OUT_DIR / "figure3_qualitative_examples.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_appendix_qualitative_gallery(generation_payload: dict) -> Path:
    out_path = OUT_DIR / "appendix_qualitative_gallery.md"
    lines = [
        "# Appendix C. Additional Generation Examples",
        "",
        "This appendix collects additional qualitative generation examples from the saved generation benchmark.",
        "",
    ]
    for dataset in generation_payload["datasets"]:
        lines.extend([f"## {dataset['display_name']}", ""])
        for run in sorted(dataset["runs"], key=lambda item: item["model_name"]):
            lines.extend(
                [
                    f"### {MODEL_SHORT.get(run['model_name'], run['model_name'])} (seed {run['seed']})",
                    "",
                ]
            )
            for idx, ex in enumerate(run["examples"], start=1):
                lines.extend(
                    [
                        f"Example {idx}",
                        "",
                        "```text",
                        f"PROMPT:\n{ex['prompt']}",
                        "",
                        f"REFERENCE:\n{ex['reference']}",
                        "",
                        f"GREEDY:\n{ex['greedy_continuation']}",
                        "",
                        f"SAMPLED:\n{ex['sampled_continuation']}",
                        "```",
                        "",
                    ]
                )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    dataset_summaries = load_dataset_summaries()
    generation_payload = load_generation_results()
    generation_avg = average_generation_metrics(generation_payload)
    averages = average_summary_metrics(dataset_summaries)

    figure1 = build_cross_dataset_loss_figure(dataset_summaries)
    figure2 = build_generation_tradeoff_figure(generation_avg)
    figure3 = build_qualitative_examples_figure(generation_payload)
    appendix_gallery = build_appendix_qualitative_gallery(generation_payload)

    metrics_path = OUT_DIR / "paper_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "average_summary_metrics": averages,
                "average_generation_metrics": generation_avg,
                "figures": [str(figure1), str(figure2), str(figure3)],
                "appendix_gallery": str(appendix_gallery),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"figure1={figure1}")
    print(f"figure2={figure2}")
    print(f"figure3={figure3}")
    print(f"appendix_gallery={appendix_gallery}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
