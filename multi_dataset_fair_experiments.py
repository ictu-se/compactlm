from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import low_resource_experiments as lre
import torch

from dataset_corpus_manager import CorpusSpec, ensure_corpus, write_dataset_manifest
from primitive_cifg_lstm_usage import PrimitiveCharCIFGLSTMLanguageModel
from primitive_gru_usage import PrimitiveCharGRULanguageModel
from primitive_lstm_usage import PrimitiveCharLSTMLanguageModel
from primitive_peephole_lstm_usage import PrimitiveCharPeepholeLSTMLanguageModel
from primitive_rnn_usage import PrimitiveCharRNNLanguageModel
from tiny_transformer import TinyTransformerDecoderLanguageModel


EXPERIMENT_NAME = "fair_matched_budget_30k_multidataset"
EXPERIMENT_DIR = Path("artifacts") / EXPERIMENT_NAME
MASTER_PROGRESS_PATH = EXPERIMENT_DIR / "progress.json"
MASTER_RESULTS_PATH = EXPERIMENT_DIR / "master_results.json"
MASTER_SUMMARY_PATH = EXPERIMENT_DIR / "master_summary.md"
CHAR_LIMIT = 1_115_394


def get_corpus_specs() -> list[CorpusSpec]:
    return [
        CorpusSpec(
            dataset_id="tinyshakespeare",
            display_name="Tiny Shakespeare",
            local_path="tinyshakespeare.txt",
        ),
        CorpusSpec(
            dataset_id="alice_in_wonderland",
            display_name="Alice in Wonderland",
            local_path="alice_in_wonderland.txt",
            url="https://www.gutenberg.org/files/11/11-0.txt",
            start_marker="*** START OF THE PROJECT GUTENBERG EBOOK 11 ***",
            end_marker="*** END OF THE PROJECT GUTENBERG EBOOK 11 ***",
            char_limit=CHAR_LIMIT,
        ),
        CorpusSpec(
            dataset_id="pride_and_prejudice",
            display_name="Pride and Prejudice",
            local_path="pride_and_prejudice.txt",
            url="https://www.gutenberg.org/files/1342/1342-0.txt",
            start_marker="*** START OF THE PROJECT GUTENBERG EBOOK 1342 ***",
            end_marker="*** END OF THE PROJECT GUTENBERG EBOOK 1342 ***",
            char_limit=CHAR_LIMIT,
        ),
        CorpusSpec(
            dataset_id="sherlock_holmes",
            display_name="The Adventures of Sherlock Holmes",
            local_path="sherlock_holmes.txt",
            url="https://www.gutenberg.org/files/1661/1661-0.txt",
            start_marker="*** START OF THE PROJECT GUTENBERG EBOOK 1661 ***",
            end_marker="*** END OF THE PROJECT GUTENBERG EBOOK 1661 ***",
            char_limit=CHAR_LIMIT,
        ),
    ]


def configure_output_paths(dataset_id: str) -> Path:
    dataset_dir = EXPERIMENT_DIR / dataset_id
    lre.ARTIFACTS_DIR = dataset_dir
    lre.FIGURES_DIR = dataset_dir / "figures"
    lre.JSON_PATH = dataset_dir / "results.json"
    lre.SUMMARY_PATH = dataset_dir / "summary.md"
    lre.PROGRESS_PATH = dataset_dir / "progress.json"
    return dataset_dir


def get_model_specs():
    return [
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


def save_master_progress(**payload) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    MASTER_PROGRESS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_master_summary(dataset_outputs: list[dict]) -> None:
    lines = [
        "# Multi-Dataset Fair Comparison Summary",
        "",
        f"Experiment: `{EXPERIMENT_NAME}`",
        "",
        "## Datasets",
        "",
        "| Dataset | Status | Summary | Results |",
        "| --- | --- | --- | --- |",
    ]
    for item in dataset_outputs:
        lines.append(
            f"| {item['display_name']} | {item['status']} | `{item['summary_path']}` | `{item['results_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Each dataset is run in its own artifact subdirectory.",
            "- Runs are resumable at the dataset/model/seed level.",
            "- Per-dataset summaries contain the detailed tables and figures.",
            "",
        ]
    )
    MASTER_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def is_cuda_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "cuda error",
            "cuda out of memory",
            "cublas",
            "cudnn",
            "device-side assert",
        )
    )


def cleanup_after_cuda_failure() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def train_single_run_with_fallback(
    *,
    model_name: str,
    family: str,
    model_factory,
    dataset: lre.PrimitiveSequenceDataset,
    seed: int,
    device: torch.device,
    corpus_spec: CorpusSpec,
    corpus_path: Path,
    dataset_dir: Path,
    completed_dataset_ids: list[str],
):
    try:
        return lre.train_single_run(
            model_name=model_name,
            family=family,
            model_factory=model_factory,
            dataset=dataset,
            seed=seed,
            device=device,
        )
    except RuntimeError as exc:
        if device.type != "cuda" or not is_cuda_runtime_error(exc):
            raise

        cleanup_after_cuda_failure()
        print(
            f"warning=cuda_failure dataset={corpus_spec.dataset_id} model={model_name} seed={seed} "
            f"retry_device=cpu error={type(exc).__name__}: {exc}",
            flush=True,
        )
        save_master_progress(
            status="retrying_seed_on_cpu",
            experiment_name=EXPERIMENT_NAME,
            device="cpu",
            original_device=str(device),
            dataset_id=corpus_spec.dataset_id,
            dataset_name=corpus_spec.display_name,
            model_name=model_name,
            seed=seed,
            dataset_path=str(corpus_path),
            dataset_artifact_dir=str(dataset_dir),
            completed_datasets=completed_dataset_ids,
            error=str(exc),
        )
        lre.save_progress(
            status="retrying_on_cpu",
            model_name=model_name,
            seed=seed,
            completed_runs=[{"model_name": run.model_name, "seed": run.seed} for run in lre.load_existing_runs()],
            extra={
                "experiment_name": EXPERIMENT_NAME,
                "dataset_id": corpus_spec.dataset_id,
                "dataset_name": corpus_spec.display_name,
                "dataset_path": str(corpus_path),
                "device": "cpu",
                "original_device": str(device),
                "error": str(exc),
            },
        )
        return lre.train_single_run(
            model_name=model_name,
            family=family,
            model_factory=model_factory,
            dataset=dataset,
            seed=seed,
            device=torch.device("cpu"),
        )


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    write_dataset_manifest(EXPERIMENT_DIR / "dataset_manifest.json", get_corpus_specs())
    device = lre.get_device()
    seeds = [7, 11, 19]
    model_specs = get_model_specs()
    dataset_outputs: list[dict] = []

    print(f"experiment={EXPERIMENT_NAME}", flush=True)
    print(f"device={device}", flush=True)

    for corpus_spec in get_corpus_specs():
        dataset_dir = configure_output_paths(corpus_spec.dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = ensure_corpus(corpus_spec)
        print(f"dataset={corpus_spec.dataset_id} path={corpus_path}", flush=True)
        dataset = lre.build_sequence_dataset(corpus_path)
        all_runs = lre.load_existing_runs()
        completed_runs = {(run.model_name, run.seed) for run in all_runs}

        save_master_progress(
            status="running_dataset",
            experiment_name=EXPERIMENT_NAME,
            device=str(device),
            dataset_id=corpus_spec.dataset_id,
            dataset_name=corpus_spec.display_name,
            dataset_path=str(corpus_path),
            dataset_artifact_dir=str(dataset_dir),
            completed_datasets=[item["dataset_id"] for item in dataset_outputs if item["status"] == "complete"],
        )

        lre.save_progress(
            status="starting",
            completed_runs=[{"model_name": run.model_name, "seed": run.seed} for run in all_runs],
            extra={
                "experiment_name": EXPERIMENT_NAME,
                "dataset_id": corpus_spec.dataset_id,
                "dataset_name": corpus_spec.display_name,
                "dataset_path": str(corpus_path),
                "device": str(device),
            },
        )

        for model_name, family, model_factory in model_specs:
            for seed in seeds:
                if (model_name, seed) in completed_runs:
                    print(
                        f"skipping completed dataset={corpus_spec.dataset_id} model={model_name} seed={seed}",
                        flush=True,
                    )
                    lre.save_progress(
                        status="skipping_completed",
                        model_name=model_name,
                        seed=seed,
                        completed_runs=[{"model_name": run.model_name, "seed": run.seed} for run in all_runs],
                        extra={
                            "experiment_name": EXPERIMENT_NAME,
                            "dataset_id": corpus_spec.dataset_id,
                            "dataset_name": corpus_spec.display_name,
                            "dataset_path": str(corpus_path),
                            "device": str(device),
                        },
                    )
                    continue

                print(
                    f"starting dataset={corpus_spec.dataset_id} model={model_name} seed={seed}",
                    flush=True,
                )
                save_master_progress(
                    status="running_seed",
                    experiment_name=EXPERIMENT_NAME,
                    device=str(device),
                    dataset_id=corpus_spec.dataset_id,
                    dataset_name=corpus_spec.display_name,
                    model_name=model_name,
                    seed=seed,
                    dataset_path=str(corpus_path),
                    dataset_artifact_dir=str(dataset_dir),
                    completed_datasets=[item["dataset_id"] for item in dataset_outputs if item["status"] == "complete"],
                )
                lre.save_progress(
                    status="starting_run",
                    model_name=model_name,
                    seed=seed,
                    completed_runs=[{"model_name": run.model_name, "seed": run.seed} for run in all_runs],
                    extra={
                        "experiment_name": EXPERIMENT_NAME,
                        "dataset_id": corpus_spec.dataset_id,
                        "dataset_name": corpus_spec.display_name,
                        "dataset_path": str(corpus_path),
                        "device": str(device),
                    },
                )
                run = train_single_run_with_fallback(
                    model_name=model_name,
                    family=family,
                    model_factory=model_factory,
                    dataset=dataset,
                    seed=seed,
                    device=device,
                    corpus_spec=corpus_spec,
                    corpus_path=corpus_path,
                    dataset_dir=dataset_dir,
                    completed_dataset_ids=[item["dataset_id"] for item in dataset_outputs if item["status"] == "complete"],
                )
                all_runs.append(run)
                completed_runs.add((model_name, seed))
                lre.JSON_PATH.write_text(
                    json.dumps(
                        {
                            "experiment_name": EXPERIMENT_NAME,
                            "dataset_id": corpus_spec.dataset_id,
                            "dataset_name": corpus_spec.display_name,
                            "dataset_path": str(corpus_path),
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
                    extra={
                        "experiment_name": EXPERIMENT_NAME,
                        "dataset_id": corpus_spec.dataset_id,
                        "dataset_name": corpus_spec.display_name,
                        "dataset_path": str(corpus_path),
                        "device": str(device),
                    },
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
                    "dataset_id": corpus_spec.dataset_id,
                    "dataset_name": corpus_spec.display_name,
                    "dataset_path": str(corpus_path),
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
                "dataset_id": corpus_spec.dataset_id,
                "dataset_name": corpus_spec.display_name,
                "dataset_path": str(corpus_path),
                "device": str(device),
                "summary_path": str(lre.SUMMARY_PATH),
                "results_path": str(lre.JSON_PATH),
            },
        )
        dataset_outputs.append(
            {
                "dataset_id": corpus_spec.dataset_id,
                "display_name": corpus_spec.display_name,
                "status": "complete",
                "summary_path": str(lre.SUMMARY_PATH),
                "results_path": str(lre.JSON_PATH),
            }
        )
        MASTER_RESULTS_PATH.write_text(json.dumps(dataset_outputs, indent=2), encoding="utf-8")
        build_master_summary(dataset_outputs)
        print(f"dataset_complete={corpus_spec.dataset_id} summary_path={lre.SUMMARY_PATH}", flush=True)

    save_master_progress(
        status="complete",
        experiment_name=EXPERIMENT_NAME,
        device=str(device),
        completed_datasets=[item["dataset_id"] for item in dataset_outputs],
        master_summary_path=str(MASTER_SUMMARY_PATH),
        master_results_path=str(MASTER_RESULTS_PATH),
    )
    print(f"master_summary_path={MASTER_SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
