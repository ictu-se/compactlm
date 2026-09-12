# Compact character-level language models

A reproducible comparison of five custom recurrent models and one small causal
Transformer on four frozen English literary corpora. The corrected protocol uses
validation-only model selection, deterministic full held-out evaluation, and the
same 64-character rolling context for every primary generation experiment.

## Environment

Validated with Python 3.9.6, PyTorch 2.8.0, NumPy 1.26.4 and Matplotlib 3.9.4 on
macOS arm64. Training explicitly uses CPU and one PyTorch thread per worker.
Other operating systems may require a compatible PyTorch wheel; exact numerical
identity across library releases and hardware is not guaranteed.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

## Reproduce the experiment

Commands resolve data paths relative to the repository, so they may be launched
from another working directory. Keep all four supplied corpus snapshots unchanged.

```sh
python3 corrected_experiments.py --output artifacts/reproduction --workers 4
```

This trains 72 models: four corpora, six model configurations, and seeds 7, 11, 19.
Each run has at most 50 epochs, 100 updates per epoch, batches of 32 sequences,
and sequence length 64. The script also generates 32 continuations per decoding
mode for every checkpoint. Runtime depends on hardware; workers consume CPU and
memory concurrently. Worker elapsed times are not comparative timing evidence.

Resume only a compatible experiment:

```sh
python3 corrected_experiments.py --output artifacts/reproduction --workers 4 --resume
```

A saved result is reused only if its configuration, computational source hash,
corpus hash and checkpoint hash agree. Missing or damaged checkpoints trigger
retraining; changes to the protocol require a new output directory. Resume restarts
an interrupted training job from its seed; it does not resume optimizer state
mid-epoch. Do not edit computational source while a run is in progress.

For a smaller diagnostic run, select an exact model label and corpus:

```sh
python3 corrected_experiments.py --output artifacts/small_run --dataset tinyshakespeare --model 'Primitive RNN Fair' --seeds 7 --workers 1
```

## Timing, integrity validation and paper assets

After all training jobs finish, run the timing benchmark **serially**, without
other experimental jobs competing for CPU:

```sh
python3 benchmark_inference.py --results artifacts/reproduction
python3 validate_results.py --results artifacts/reproduction
python3 build_paper_results_assets.py --results artifacts/reproduction --output artifacts/reproduced_assets
```

The timing benchmark measures batch-one greedy generation on Tiny Shakespeare,
with 64 prompt characters and 128 new characters. Each measurement uses one
warmup and three timed repetitions. Fixed-window recomputation and recurrent
full-history state reuse have different context semantics and are reported
separately. Neither is a claim about optimized GPU or edge-device performance.

The asset builder refuses incomplete sets of runs and verifies checkpoint hashes,
true-minimum selection and held-out target counts before creating tables and
figures. `validate_results.py` additionally loads all 72 checkpoints, recomputes
held-out losses and generation diagnostics, and checks prompt identity across
models. The generated LaTeX fragments contain experiment-derived tables and
numeric macros only; manuscript source and manuscript PDFs are kept outside this
repository. Figures are generated exclusively from measured results.

## Protocol and interpretation

- Sequential 90/5/5 character splits, using the complete corpus's sorted alphabet.
  This is an explicit closed-alphabet assumption. One held-out Sherlock character
  does not occur in training.
- Training batches have an independent random generator. Model initialization,
  dropout, validation and generation do not alter the training-window sequence.
- Validation and test evaluate every target except the first exactly once using
  disjoint blocks of at most 64 characters, resetting model state at each block.
- The best checkpoint is the strict minimum validation loss. A separate 0.002-nat
  tolerance and five-epoch patience control stopping. Reaching the 50-epoch cap is
  recorded as budget exhaustion, not convergence. Test loss is computed only after
  checkpoint selection.
- Generation uses 32 nonoverlapping held-out prompt/reference blocks per corpus,
  shared across all models and seeds. Prompts have 64 characters; continuations
  have 160. Both greedy and temperature-0.8 sampling use the last 64 characters.
- Trigram overlap and four-gram repetition are surface diagnostics, not semantic
  quality or factuality measures. The three training seeds are the replicates;
  prompts are averaged within checkpoints, not counted as independent models.
- Budgets are approximately matched, not equal. Width, gate implementation,
  Transformer dropout, and architecture differ together. No architecture-specific
  tuning was performed. Conclusions are limited to these configurations and texts.
- Custom GRU reset gating occurs before the recurrent candidate matrix product;
  substituting a library GRU can change its equations.

## Replication contents

`artifacts/corrected_v1` holds the corrected configuration, run index, checkpoints,
training histories, final metrics and all generated continuations. Each run records
source/data/checkpoint hashes and its execution environment. `artifacts/corrected_assets`
contains aggregate measurements and generated figures. Table fragments are regenerated
locally by the asset builder and are not tracked in Git. The original,
methodologically different results are preserved outside this working replication
package and remain recoverable in the original Git history; do not pool them with
corrected results.

Corpus provenance: [char-rnn Tiny Shakespeare](https://github.com/karpathy/char-rnn),
[Project Gutenberg 11](https://www.gutenberg.org/ebooks/11),
[1342](https://www.gutenberg.org/ebooks/1342), and
[1661](https://www.gutenberg.org/ebooks/1661). The Gutenberg corpus manager documents
the original extraction boundaries. Training uses frozen copies rather than
silently downloading a potentially changed edition. Consult the original sources
for text provenance and applicable redistribution terms.

## Verify the published results without retraining

```sh
python3 validate_results.py --results artifacts/corrected_v1
python3 build_paper_results_assets.py --results artifacts/corrected_v1 --output artifacts/reproduced_assets
```

The recorded computational source SHA256 is
`20fdba9a3f29555f63cdd9dfffeddd266c8f3bcd80bab5faa81b796710873772`.
It covers the training protocol and model implementations; corpus and checkpoint
hashes are recorded separately in each run. Historical results can be recovered
from commit `c9042ece7f079e848e8a79239526d81d07686bd1`.
