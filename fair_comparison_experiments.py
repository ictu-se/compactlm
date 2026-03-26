import json
from pathlib import Path

import low_resource_experiments as lre

from primitive_cifg_lstm_usage import PrimitiveCharCIFGLSTMLanguageModel
from primitive_gru_usage import PrimitiveCharGRULanguageModel
from primitive_lstm_usage import PrimitiveCharLSTMLanguageModel
from primitive_peephole_lstm_usage import PrimitiveCharPeepholeLSTMLanguageModel
from primitive_rnn_usage import PrimitiveCharRNNLanguageModel
from tiny_transformer import TinyTransformerDecoderLanguageModel


EXPERIMENT_NAME = "fair_matched_budget_30k"
EXPERIMENT_DIR = Path("artifacts") / EXPERIMENT_NAME


def configure_output_paths() -> None:
    lre.ARTIFACTS_DIR = EXPERIMENT_DIR
    lre.FIGURES_DIR = EXPERIMENT_DIR / "figures"
    lre.JSON_PATH = EXPERIMENT_DIR / "results.json"
    lre.SUMMARY_PATH = EXPERIMENT_DIR / "summary.md"
    lre.PROGRESS_PATH = EXPERIMENT_DIR / "progress.json"


def main() -> None:
    configure_output_paths()
    lre.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = lre.get_device()
    print(f"experiment={EXPERIMENT_NAME}", flush=True)
    print(f"device={device}", flush=True)
    dataset = lre.build_sequence_dataset(Path("tinyshakespeare.txt"))
    seeds = [7, 11, 19]

    # Each family is adjusted to land near the same parameter budget (~30k trainable params).
    model_specs = [
        (
            "Primitive RNN Fair",
            "RNN",
            lambda vocab_size: PrimitiveCharRNNLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=125),
        ),
        (
            "Primitive GRU Fair",
            "GRU",
            lambda vocab_size: PrimitiveCharGRULanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=73),
        ),
        (
            "Primitive LSTM Fair",
            "LSTM",
            lambda vocab_size: PrimitiveCharLSTMLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=62),
        ),
        (
            "Primitive Peephole LSTM Fair",
            "LSTM Variant",
            lambda vocab_size: PrimitiveCharPeepholeLSTMLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=62),
        ),
        (
            "Primitive CIFG LSTM Fair",
            "LSTM Variant",
            lambda vocab_size: PrimitiveCharCIFGLSTMLanguageModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=73),
        ),
        (
            "Tiny Transformer Decoder Fair",
            "Transformer",
            lambda vocab_size: TinyTransformerDecoderLanguageModel(
                vocab_size=vocab_size,
                d_model=48,
                num_heads=4,
                ff_dim=112,
                num_layers=1,
                max_seq_len=64,
                dropout=0.1,
            ),
        ),
    ]

    all_runs = lre.load_existing_runs()
    completed_runs = {(run.model_name, run.seed) for run in all_runs}
    lre.save_progress(
        status="starting",
        completed_runs=[{"model_name": run.model_name, "seed": run.seed} for run in all_runs],
        extra={"experiment_name": EXPERIMENT_NAME, "device": str(device)},
    )
    for model_name, family, model_factory in model_specs:
        for seed in seeds:
            if (model_name, seed) in completed_runs:
                print(f"skipping completed model={model_name} seed={seed}", flush=True)
                lre.save_progress(
                    status="skipping_completed",
                    model_name=model_name,
                    seed=seed,
                    completed_runs=[{"model_name": run.model_name, "seed": run.seed} for run in all_runs],
                    extra={"experiment_name": EXPERIMENT_NAME, "device": str(device)},
                )
                continue
            print(f"starting model={model_name} seed={seed}", flush=True)
            lre.save_progress(
                status="starting_run",
                model_name=model_name,
                seed=seed,
                completed_runs=[{"model_name": run.model_name, "seed": run.seed} for run in all_runs],
                extra={"experiment_name": EXPERIMENT_NAME, "device": str(device)},
            )
            run = lre.train_single_run(
                model_name=model_name,
                family=family,
                model_factory=model_factory,
                dataset=dataset,
                seed=seed,
                device=device,
            )
            all_runs.append(run)
            completed_runs.add((model_name, seed))
            lre.JSON_PATH.write_text(
                json.dumps(
                    {
                        "experiment_name": EXPERIMENT_NAME,
                        "runs": [lre.serialize_run_metrics(item) for item in all_runs],
                        "summaries": lre.summarize_runs(all_runs),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            lre.save_progress(
                status="saved_completed_run",
                model_name=model_name,
                seed=seed,
                epoch=run.best_epoch,
                best_epoch=run.best_epoch,
                best_val_loss=run.best_val_loss,
                latest_test_loss=run.final_test_loss,
                completed_runs=[{"model_name": item.model_name, "seed": item.seed} for item in all_runs],
                extra={"experiment_name": EXPERIMENT_NAME, "device": str(device)},
            )

    summaries = lre.summarize_runs(all_runs)
    figure_paths = [
        lre.plot_quality_efficiency(summaries),
        lre.plot_learning_curves(summaries),
    ]
    lre.write_summary_markdown(summaries, figure_paths)
    lre.JSON_PATH.write_text(
        json.dumps(
            {
                "experiment_name": EXPERIMENT_NAME,
                "runs": [lre.serialize_run_metrics(item) for item in all_runs],
                "summaries": summaries,
                "figures": [str(path) for path in figure_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lre.save_progress(
        status="complete",
        completed_runs=[{"model_name": item.model_name, "seed": item.seed} for item in all_runs],
        extra={
            "experiment_name": EXPERIMENT_NAME,
            "device": str(device),
            "summary_path": str(lre.SUMMARY_PATH),
            "results_path": str(lre.JSON_PATH),
        },
    )
    print(f"summary_path={lre.SUMMARY_PATH}", flush=True)
    for figure_path in figure_paths:
        print(f"figure_path={figure_path}", flush=True)


if __name__ == "__main__":
    main()
