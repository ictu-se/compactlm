# Controlled follow-up protocol

This follow-up was designed after the original comparison was inspected. It is
not a preregistration of the original study. The machine-readable plan is frozen
before any new ablation outcomes are generated. The test set is not used to select
new configurations or to expand the grid in response to outcomes.

## Interventions

The Transformer grid crosses all combinations of:

- Dropout probability: 0.0 and 0.1, at the original embedding, attention-probability
  and feed-forward dropout locations.
- Adam learning rate: 0.0003, 0.001 and 0.003; all other optimizer settings unchanged.
- Allocation: original model width 48/feed-forward width 112, or model width 32
  with feed-forward width determined solely by matching the original parameter
  count on each corpus. The alternative widths are 296, 301, 309 and 308 for
  Shakespeare, Alice, Pride and Sherlock, respectively. Parameter-count differences
  are below 0.1%. All configurations retain four heads and one block.

This is a 2 x 3 x 2 factorial grid, not a collection of configurations selected
because they happened to perform well. The allocation intervention changes the
embedding, attention and feed-forward dimensions together under a common budget;
it does not identify a separate causal effect of any one component.

RNN and GRU each receive the same three learning rates, with their original widths
and no dropout. They provide learning-rate sensitivity controls, not an equal-size
architecture-search budget. All settings use four frozen corpora and seeds 7, 11,
19. There are 216 combinations, including 36 original combinations reused after
hash and validation checks; 180 combinations require new training.

## Training and separation of selection from test evaluation

Training retains the original batch size 32, sequence length 64, Adam optimizer,
100 updates per epoch, maximum 50 epochs, validation tolerance 0.002 and patience
five. The true validation minimum determines checkpoint saving independently of
the patience tolerance. All comparisons use identical training-window sequences
for a given corpus and seed. Within an architecture/allocation, dropout and
learning-rate interventions also share identical initialized weights.

Training records contain validation loss and history, but no test scores. Only
after all 216 training records are complete does the program choose one global
configuration per architecture by validation NLL averaged equally over all four
corpora and three seeds. Ties are resolved by configuration identifier. This choice
and the hashes of the immutable training records are saved before test evaluation.
No outcome-dependent extension of the grid is allowed in this experiment.

All settings then receive full deterministic training, validation and test NLL
evaluation at their selected checkpoint. These scores reset context at the same
64-character block boundaries. Training scores are computed in evaluation mode,
so they are not inflated by active dropout. All test targets except the first are
scored exactly once. Training scores and held-out gaps are diagnostics, not an
independent identification of overfitting or data-distribution effects.

Greedy and temperature-0.8 generation are additionally evaluated for the globally
validation-selected setting of each architecture: 32 nonoverlapping prompts,
64-character context and 160 generated characters, as in the original protocol.
This adds 2,304 continuations across the 36 selected checkpoints. It does not add
a temperature or sampling-strategy sweep.

## Planned analysis

Report all twelve Transformer settings, including their training, validation and
test losses; do not report only the best. Compute paired test-NLL contrasts for:

1. Dropout 0.0 minus 0.1, averaging over allocation and learning rate.
2. Learning rate 0.0003 minus 0.001, averaging over dropout and allocation.
3. Learning rate 0.003 minus 0.001, averaging over dropout and allocation.
4. Width-32 allocation minus width-48 allocation, averaging over dropout and rate.

Negative contrasts favor the first-listed setting. Compute contrasts within each
corpus and training seed before averaging. Show corpus-specific effects and sample
standard deviations across the three seed summaries. No claim of statistical
significance or a universal architectural cause is planned. Conditional contrasts
and difference-in-differences quantify interactions when a marginal effect hides
sensitivity to the other factors.

Report the original and globally validation-selected settings for Transformer,
RNN and GRU, with actual parameters and stopping counts. The Transformer receives
more configurations than the recurrent controls, so this is a diagnostic follow-up,
not a claim of equal tuning effort. Effects are interventions under the stated
training/stopping/selection policy; differences in stopping time can mediate them.

## Commands

```sh
python3 -m unittest discover -s tests -v
python3 ablation_experiments.py --phase freeze
python3 ablation_experiments.py --workers 4
python3 validate_ablation.py
python3 build_ablation_assets.py
```

Re-running the command resumes compatible completed records and retrains any
missing-checkpoint job. Do not change computational sources during a run. Use a
new output directory for any changed plan. The original experiment and its source
hash remain intact; ablation code has an additional separate source hash.
