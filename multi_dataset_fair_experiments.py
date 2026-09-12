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



def main():
    from corrected_experiments import main as run
    run()

if __name__ == "__main__":
    main()
