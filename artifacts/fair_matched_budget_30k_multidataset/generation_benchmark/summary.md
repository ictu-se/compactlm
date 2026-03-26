# Generation Quality Benchmark

This report evaluates free-running continuation quality from saved checkpoints.

## Protocol

- Prompts per dataset: `4`
- Prompt length: `96` chars
- Continuation length: `160` chars
- Sample temperature: `0.8`
- Evaluated seeds: `best validation seed per model`

## Metrics

- `char_accuracy`: exact character match rate against the held-out continuation.
- `prefix_match_ratio`: how long generation stays on the reference before diverging.
- `char_trigram_f1` / `char_fourgram_f1`: local text overlap with the held-out continuation.
- `repetition_rate_4gram`: fraction of repeated character 4-grams inside generated text.
- `distinct_word_bigram_ratio`: surface diversity at the word level.

## Tiny Shakespeare

| Model | Seed | Test Loss | Greedy char-4 F1 | Greedy repetition | Sampled char-4 F1 | Sampled repetition | Gen chars/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primitive GRU Fair | 19 | 1.8083 | 0.0127 | 0.9315 | 0.0701 | 0.0494 | 30.64 |
| Primitive LSTM Fair | 11 | 1.8771 | 0.0111 | 0.9108 | 0.0510 | 0.0446 | 21.46 |
| Primitive RNN Fair | 11 | 1.8466 | 0.0064 | 0.9395 | 0.0462 | 0.0478 | 107.26 |
| Primitive Peephole LSTM Fair | 19 | 1.8312 | 0.0048 | 0.8583 | 0.0541 | 0.0653 | 14.36 |
| Primitive CIFG LSTM Fair | 7 | 1.8730 | 0.0000 | 0.9188 | 0.0239 | 0.0462 | 20.08 |
| Tiny Transformer Decoder Fair | 11 | 2.0277 | 0.0000 | 0.9697 | 0.0573 | 0.0780 | 1160.46 |

Best greedy char-4 F1: `Primitive GRU Fair` (seed `19`).

Example prompt/reference/generated continuation:

```text
PROMPT:
rance ta'en
As shall with either part's agreement stand?

BAPTISTA:
Not in my house, Lucentio; f

REFERENCE:
or, you know,
Pitchers have ears, and I have many servants:
Besides, old Gremio is hearkening still;
And happily we might be interrupted.

TRANIO:
Then at my lo

GREEDY:
or the shall the shall the shall the shall the shall the shall the shall the shall the shall the shall the shall the shall the shall the shall the shall the sha

SAMPLED:
or unding by anpeate and have dreast me, and should too part,
And my heart, you is the hands,
When nis the fanger good dead.

SICINIUS:
At with but an may such 
```

## Alice in Wonderland

| Model | Seed | Test Loss | Greedy char-4 F1 | Greedy repetition | Sampled char-4 F1 | Sampled repetition | Gen chars/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primitive RNN Fair | 11 | 1.6257 | 0.0669 | 0.6481 | 0.0939 | 0.0717 | 71.83 |
| Primitive CIFG LSTM Fair | 19 | 1.6238 | 0.0621 | 0.8010 | 0.0908 | 0.0844 | 28.97 |
| Tiny Transformer Decoder Fair | 19 | 1.7851 | 0.0494 | 0.9061 | 0.1178 | 0.1306 | 1475.81 |
| Primitive GRU Fair | 7 | 1.6373 | 0.0414 | 0.8487 | 0.0939 | 0.0892 | 22.30 |
| Primitive LSTM Fair | 19 | 1.6664 | 0.0382 | 0.8121 | 0.1035 | 0.0892 | 25.56 |
| Primitive Peephole LSTM Fair | 7 | 1.6625 | 0.0350 | 0.8567 | 0.1003 | 0.0780 | 20.77 |

Best greedy char-4 F1: `Primitive RNN Fair` (seed `11`).

Example prompt/reference/generated continuation:

```text
PROMPT:
me like an honest man.”

There was a general clapping of hands at this: it was the first really


REFERENCE:
clever thing the King had said that day.

“That _proves_ his guilt,” said the Queen.

“It proves nothing of the sort!” said Alice. “Why, you don’t even know
wha

GREEDY:
said to herself, “and the boots to herself, “and the boots to herself, “and the boots to herself, “and the boots to herself, “and the boots to herself, “and the

SAMPLED:
give peep and the mather

with the and foun was extifes
at Alice veifucl’s holy was something a patter.)

“re to mome that a rather till who every got to went m
```

## Pride and Prejudice

| Model | Seed | Test Loss | Greedy char-4 F1 | Greedy repetition | Sampled char-4 F1 | Sampled repetition | Gen chars/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primitive RNN Fair | 19 | 1.4582 | 0.1465 | 0.8185 | 0.1576 | 0.2182 | 112.09 |
| Primitive LSTM Fair | 7 | 1.6007 | 0.1322 | 0.8041 | 0.0924 | 0.1210 | 25.87 |
| Primitive CIFG LSTM Fair | 7 | 1.5347 | 0.1226 | 0.8201 | 0.0669 | 0.0653 | 24.30 |
| Primitive GRU Fair | 7 | 1.4755 | 0.1178 | 0.8360 | 0.1497 | 0.1704 | 30.71 |
| Primitive Peephole LSTM Fair | 11 | 1.6038 | 0.1131 | 0.8742 | 0.1624 | 0.1656 | 20.68 |
| Tiny Transformer Decoder Fair | 19 | 1.8114 | 0.1035 | 0.9475 | 0.1306 | 0.4395 | 1198.52 |

Best greedy char-4 F1: `Primitive RNN Fair` (seed `19`).

Example prompt/reference/generated continuation:

```text
PROMPT:
lterable, they are not, I hope, quite so easily
changed as that implies.”

“When I wrote that le

REFERENCE:
tter,” replied Darcy, “I believed myself perfectly
calm and cool; but I am since convinced that it was written in a
dreadful bitterness of spirit.”

“The letter

GREEDY:
ave the good and the sure you was the good and the sure you was the good and the sure you was the good and the sure you was the good and the sure you was the go

SAMPLED:
ast hes. You_ against of dourh to her certaith wishes of sligation that whot she was reaport to her time to the disapcersame very are soon anxious to conceity c
```

## The Adventures of Sherlock Holmes

| Model | Seed | Test Loss | Greedy char-4 F1 | Greedy repetition | Sampled char-4 F1 | Sampled repetition | Gen chars/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primitive GRU Fair | 19 | 1.6137 | 0.0764 | 0.7054 | 0.0876 | 0.0573 | 20.89 |
| Primitive Peephole LSTM Fair | 7 | 1.6890 | 0.0669 | 0.7468 | 0.0764 | 0.0732 | 15.87 |
| Primitive LSTM Fair | 11 | 1.6952 | 0.0414 | 0.8901 | 0.1067 | 0.0525 | 17.13 |
| Primitive CIFG LSTM Fair | 11 | 1.6314 | 0.0366 | 0.8471 | 0.0844 | 0.0669 | 19.77 |
| Primitive RNN Fair | 11 | 1.6007 | 0.0334 | 0.8838 | 0.0812 | 0.0685 | 72.64 |
| Tiny Transformer Decoder Fair | 7 | 1.8661 | 0.0303 | 0.9475 | 0.0971 | 0.1115 | 1191.90 |

Best greedy char-4 F1: `Primitive GRU Fair` (seed `19`).

Example prompt/reference/generated continuation:

```text
PROMPT:
 nor the reverse. She was a
nonentity. It was easy to see that she was passionately devoted both

REFERENCE:
 to
her husband and to her little son. Her light grey eyes wandered
continually from one to the other, noting every little want and
forestalling it if possible.

GREEDY:
 and some the could not that I have been that I have been that I have been that I have been that I have been that I have been that I have been that I have been 

SAMPLED:
 the can among and all net train and did me getsed only that you have beturient with groose, in a cruch of his marked, which would more and had perhad had a sha
```
