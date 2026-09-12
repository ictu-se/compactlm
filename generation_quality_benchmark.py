"""Surface diagnostics; these metrics do not measure semantic quality."""
from __future__ import annotations
import re
from collections import Counter

def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0

def char_accuracy(generated: str, reference: str) -> float:
    return safe_div(sum(a == b for a, b in zip(generated, reference)), min(len(generated), len(reference)))

def longest_common_prefix_ratio(generated: str, reference: str) -> float:
    count = 0
    for a, b in zip(generated, reference):
        if a != b:
            break
        count += 1
    return safe_div(count, min(len(generated), len(reference)))

def char_ngram_counter(text: str, n: int) -> Counter[str]:
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))

def char_ngram_f1(generated: str, reference: str, n: int) -> float:
    gen_counts = char_ngram_counter(generated, n)
    ref_counts = char_ngram_counter(reference, n)
    if not gen_counts or not ref_counts:
        return 0.0
    overlap = sum((gen_counts & ref_counts).values())
    precision = safe_div(overlap, sum(gen_counts.values()))
    recall = safe_div(overlap, sum(ref_counts.values()))
    return safe_div(2 * precision * recall, precision + recall)

def repetition_rate(text: str, n: int) -> float:
    counts = char_ngram_counter(text, n)
    total = sum(counts.values())
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return safe_div(repeated, total)

def distinct_word_bigram_ratio(text: str) -> float:
    words = re.findall(r"\S+", text)
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return safe_div(len(set(bigrams)), len(bigrams))

def evaluate_generation(generated: str, reference: str) -> dict[str, float]:
    return {
        "char_accuracy": char_accuracy(generated, reference),
        "prefix_match_ratio": longest_common_prefix_ratio(generated, reference),
        "char_trigram_f1": char_ngram_f1(generated, reference, 3),
        "char_fourgram_f1": char_ngram_f1(generated, reference, 4),
        "repetition_rate_4gram": repetition_rate(generated, 4),
        "distinct_word_bigram_ratio": distinct_word_bigram_ratio(generated),
    }

if __name__=='__main__':
    raise SystemExit('Generation is now included in corrected_experiments.py for every seed. See README.md.')
