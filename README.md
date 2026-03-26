# Reviewer Package

This folder is a compact replication package for the experiments reported in the paper. It contains the minimum code, datasets, results, and figures needed for reviewers to inspect the claims and, if desired, rerun the core experiments.

## What is included
- Core experiment scripts:
  - `fair_comparison_experiments.py`
  - `multi_dataset_fair_experiments.py`
  - `generation_quality_benchmark.py`
  - `build_paper_results_assets.py`
- Minimal code dependencies required by those scripts:
  - `low_resource_experiments.py`
  - `dataset_corpus_manager.py`
  - `primitive_usage_helpers.py`
  - `tiny_transformer.py`
  - `primitive_rnn.py`, `primitive_rnn_usage.py`
  - `primitive_gru.py`, `primitive_gru_usage.py`
  - `primitive_lstm.py`, `primitive_lstm_usage.py`
  - `primitive_peephole_lstm.py`, `primitive_peephole_lstm_usage.py`
  - `primitive_cifg_lstm.py`, `primitive_cifg_lstm_usage.py`
- Input corpora:
  - `tinyshakespeare.txt`
  - `datasets/alice_in_wonderland.txt`
  - `datasets/pride_and_prejudice.txt`
  - `datasets/sherlock_holmes.txt`
- Output artifacts:
  - `artifacts/fair_matched_budget_30k/`
  - `artifacts/fair_matched_budget_30k_multidataset/`
  - `artifacts/paper_figures/`

## What is intentionally excluded
- Manuscript files and bibliography.
- Model checkpoint files (`*.pt`) to keep the package lightweight.
- Large logs and unrelated exploratory artifacts.

## Environment
Recommended environment:
- Python 3.12+
- PyTorch 2.10+
- matplotlib
- numpy

A minimal install example is:

```bash
pip install torch matplotlib numpy
```

If you want GPU execution, install the PyTorch build that matches your CUDA setup.

## Folder assumptions
- `fair_comparison_experiments.py` expects `tinyshakespeare.txt` in the package root.
- `multi_dataset_fair_experiments.py` and `generation_quality_benchmark.py` expect the other corpora under `datasets/`.
- All generated outputs are written under `artifacts/`.

## How to rerun the core experiments

### 1. Single-dataset matched-budget fair comparison
This reproduces the main matched-budget experiment on Tiny Shakespeare.

```bash
python fair_comparison_experiments.py
```

Outputs:
- `artifacts/fair_matched_budget_30k/results.json`
- `artifacts/fair_matched_budget_30k/summary.md`
- `artifacts/fair_matched_budget_30k/figures/`

### 2. Multi-dataset matched-budget benchmark
This reproduces the four-dataset extension.

```bash
python multi_dataset_fair_experiments.py
```

Outputs:
- `artifacts/fair_matched_budget_30k_multidataset/master_results.json`
- `artifacts/fair_matched_budget_30k_multidataset/master_summary.md`
- per-dataset `results.json`, `summary.md`, and `figures/`

### 3. Free-running generation benchmark
This evaluates saved best-run summaries using the generation metrics reported in the paper.

```bash
python generation_quality_benchmark.py
```

Outputs:
- `artifacts/fair_matched_budget_30k_multidataset/generation_benchmark/results.json`
- `artifacts/fair_matched_budget_30k_multidataset/generation_benchmark/summary.md`

Important note:
- Full rerunning of the generation benchmark from trained weights normally requires checkpoint files.
- In this lightweight package, checkpoint files are intentionally excluded.
- The included `generation_benchmark` outputs are therefore the main reviewer-facing evidence unless checkpoints are restored separately.

### 4. Rebuild the paper figures
This regenerates the final figure assets from the included JSON outputs.

```bash
python build_paper_results_assets.py
```

Outputs:
- `artifacts/paper_figures/figure1_cross_dataset_test_loss.png`
- `artifacts/paper_figures/figure2_generation_tradeoff.png`
- `artifacts/paper_figures/figure3_qualitative_examples.png`
- `artifacts/paper_figures/appendix_qualitative_gallery.md`
- `artifacts/paper_figures/paper_metrics.json`

## Which files matter most for reviewer inspection
If a reviewer only wants to verify the paper claims without rerunning everything, the most important files are:
- `artifacts/fair_matched_budget_30k/summary.md`
- `artifacts/fair_matched_budget_30k/results.json`
- `artifacts/fair_matched_budget_30k_multidataset/master_summary.md`
- `artifacts/fair_matched_budget_30k_multidataset/master_results.json`
- `artifacts/fair_matched_budget_30k_multidataset/generation_benchmark/summary.md`
- `artifacts/fair_matched_budget_30k_multidataset/generation_benchmark/results.json`
- `artifacts/paper_figures/`

## Practical limitation
This package is designed to be small enough for GitHub review. It is therefore sufficient for inspecting outputs and reproducing figure generation, but not for fully retraining-and-regenerating every benchmark end-to-end unless model checkpoints are added back.
