from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

import low_resource_experiments as lre
from dataset_corpus_manager import ensure_corpus
from multi_dataset_fair_experiments import get_corpus_specs, get_model_specs
from primitive_usage_helpers import PrimitiveSequenceDataset, build_sequence_dataset


DEFAULT_EXPERIMENT_DIR = Path("artifacts") / "fair_matched_budget_30k_multidataset"


@dataclass
class PromptCase:
    prompt: str
    reference: str
    start_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark free-running generation quality for saved checkpoints.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Directory containing per-dataset results.json files.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        help="Optional dataset id filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Evaluate every saved seed instead of the best validation seed per model.",
    )
    parser.add_argument("--prompt-length", type=int, default=96)
    parser.add_argument("--continuation-length", type=int, default=200)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return lre.get_device()


def sanitize_filename(name: str) -> str:
    return name.lower().replace(" ", "_")


def select_prompt_cases(
    dataset: PrimitiveSequenceDataset,
    prompt_length: int,
    continuation_length: int,
    num_prompts: int,
) -> list[PromptCase]:
    test_text = "".join(dataset.itos[int(idx)] for idx in dataset.test_tokens.tolist())
    window = prompt_length + continuation_length
    if len(test_text) <= window + 1:
        raise ValueError("Test split is too short for the requested prompt/continuation lengths.")

    max_start = len(test_text) - window
    if num_prompts == 1:
        starts = [max_start // 2]
    else:
        step = max(1, max_start // (num_prompts - 1))
        starts = [min(i * step, max_start) for i in range(num_prompts)]

    cases: list[PromptCase] = []
    for start in starts:
        prompt = test_text[start : start + prompt_length]
        reference = test_text[start + prompt_length : start + window]
        cases.append(PromptCase(prompt=prompt, reference=reference, start_index=start))
    return cases


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


def average_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys()
    return {key: statistics.mean(item[key] for item in items) for key in keys}


def get_dataset_results_paths(experiment_dir: Path, dataset_filter: set[str] | None) -> list[tuple[str, Path]]:
    if (experiment_dir / "master_results.json").exists():
        payload = json.loads((experiment_dir / "master_results.json").read_text(encoding="utf-8"))
        outputs = []
        for item in payload:
            dataset_id = item["dataset_id"]
            if dataset_filter and dataset_id not in dataset_filter:
                continue
            outputs.append((dataset_id, Path(item["results_path"])))
        return outputs

    results_path = experiment_dir / "results.json"
    if results_path.exists():
        dataset_id = experiment_dir.name
        if dataset_filter and dataset_id not in dataset_filter:
            return []
        return [(dataset_id, results_path)]
    raise FileNotFoundError(f"Could not find results metadata under {experiment_dir}")


def choose_runs(payload: dict, all_seeds: bool) -> list[dict]:
    runs = payload.get("runs", [])
    if all_seeds:
        return runs
    chosen: dict[str, dict] = {}
    for run in runs:
        current = chosen.get(run["model_name"])
        if current is None or (run["best_val_loss"], run["seed"]) < (current["best_val_loss"], current["seed"]):
            chosen[run["model_name"]] = run
    return list(chosen.values())


def build_model_factories() -> dict[str, tuple[str, object]]:
    return {name: (family, factory) for name, family, factory in get_model_specs()}


def make_model(model_name: str, dataset: PrimitiveSequenceDataset):
    factories = build_model_factories()
    if model_name not in factories:
        raise KeyError(f"No factory found for model {model_name}")
    _, factory = factories[model_name]
    return factory(len(dataset.stoi))


def benchmark_run(
    *,
    run_payload: dict,
    dataset: PrimitiveSequenceDataset,
    device: torch.device,
    checkpoint_path: Path,
    prompt_cases: list[PromptCase],
    sample_temperature: float,
) -> dict:
    model = make_model(run_payload["model_name"], dataset)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    greedy_scores = []
    sampled_scores = []
    prompt_outputs = []

    started = time.perf_counter()
    for case in prompt_cases:
        greedy_text = lre.generate_text_on_device(
            model,
            dataset,
            device,
            prompt=case.prompt,
            max_new_tokens=len(case.reference),
            temperature=None,
        )
        sampled_text = lre.generate_text_on_device(
            model,
            dataset,
            device,
            prompt=case.prompt,
            max_new_tokens=len(case.reference),
            temperature=sample_temperature,
        )
        greedy_continuation = greedy_text[len(case.prompt) :]
        sampled_continuation = sampled_text[len(case.prompt) :]

        greedy_scores.append(evaluate_generation(greedy_continuation, case.reference))
        sampled_scores.append(evaluate_generation(sampled_continuation, case.reference))
        prompt_outputs.append(
            {
                "start_index": case.start_index,
                "prompt": case.prompt,
                "reference": case.reference,
                "greedy_continuation": greedy_continuation,
                "sampled_continuation": sampled_continuation,
            }
        )

    elapsed = max(time.perf_counter() - started, 1e-9)
    generated_chars = len(prompt_cases) * len(prompt_cases[0].reference) * 2
    return {
        "model_name": run_payload["model_name"],
        "seed": run_payload["seed"],
        "best_val_loss": run_payload["best_val_loss"],
        "final_test_loss": run_payload["final_test_loss"],
        "greedy_metrics": average_metric_dict(greedy_scores),
        "sampled_metrics": average_metric_dict(sampled_scores),
        "generation_chars_per_second": generated_chars / elapsed,
        "examples": prompt_outputs[: min(3, len(prompt_outputs))],
    }


def write_outputs(experiment_dir: Path, benchmark_payload: dict) -> tuple[Path, Path]:
    output_dir = experiment_dir / "generation_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(benchmark_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Generation Quality Benchmark",
        "",
        "This report evaluates free-running continuation quality from saved checkpoints.",
        "",
        "## Protocol",
        "",
        f"- Prompts per dataset: `{benchmark_payload['config']['num_prompts']}`",
        f"- Prompt length: `{benchmark_payload['config']['prompt_length']}` chars",
        f"- Continuation length: `{benchmark_payload['config']['continuation_length']}` chars",
        f"- Sample temperature: `{benchmark_payload['config']['sample_temperature']}`",
        f"- Evaluated seeds: `{'all' if benchmark_payload['config']['all_seeds'] else 'best validation seed per model'}`",
        "",
        "## Metrics",
        "",
        "- `char_accuracy`: exact character match rate against the held-out continuation.",
        "- `prefix_match_ratio`: how long generation stays on the reference before diverging.",
        "- `char_trigram_f1` / `char_fourgram_f1`: local text overlap with the held-out continuation.",
        "- `repetition_rate_4gram`: fraction of repeated character 4-grams inside generated text.",
        "- `distinct_word_bigram_ratio`: surface diversity at the word level.",
        "",
    ]

    for dataset_result in benchmark_payload["datasets"]:
        lines.extend(
            [
                f"## {dataset_result['display_name']}",
                "",
                "| Model | Seed | Test Loss | Greedy char-4 F1 | Greedy repetition | Sampled char-4 F1 | Sampled repetition | Gen chars/s |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        rows = sorted(
            dataset_result["runs"],
            key=lambda item: item["greedy_metrics"]["char_fourgram_f1"],
            reverse=True,
        )
        for row in rows:
            lines.append(
                f"| {row['model_name']} | {row['seed']} | {row['final_test_loss']:.4f} | "
                f"{row['greedy_metrics']['char_fourgram_f1']:.4f} | {row['greedy_metrics']['repetition_rate_4gram']:.4f} | "
                f"{row['sampled_metrics']['char_fourgram_f1']:.4f} | {row['sampled_metrics']['repetition_rate_4gram']:.4f} | "
                f"{row['generation_chars_per_second']:.2f} |"
            )
        lines.append("")
        best_row = rows[0]
        example = best_row["examples"][0]
        lines.extend(
            [
                f"Best greedy char-4 F1: `{best_row['model_name']}` (seed `{best_row['seed']}`).",
                "",
                "Example prompt/reference/generated continuation:",
                "",
                "```text",
                f"PROMPT:\n{example['prompt']}",
                "",
                f"REFERENCE:\n{example['reference']}",
                "",
                f"GREEDY:\n{example['greedy_continuation']}",
                "",
                f"SAMPLED:\n{example['sampled_continuation']}",
                "```",
                "",
            ]
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dataset_filter = set(args.dataset_ids) if args.dataset_ids else None
    results_paths = get_dataset_results_paths(args.experiment_dir, dataset_filter)
    corpus_specs = {spec.dataset_id: spec for spec in get_corpus_specs()}

    payload = {
        "experiment_dir": str(args.experiment_dir),
        "device": str(device),
        "config": {
            "num_prompts": args.num_prompts,
            "prompt_length": args.prompt_length,
            "continuation_length": args.continuation_length,
            "sample_temperature": args.sample_temperature,
            "all_seeds": args.all_seeds,
        },
        "datasets": [],
    }

    for dataset_id, results_path in results_paths:
        spec = corpus_specs[dataset_id]
        corpus_path = ensure_corpus(spec)
        dataset = build_sequence_dataset(corpus_path)
        prompt_cases = select_prompt_cases(
            dataset,
            prompt_length=args.prompt_length,
            continuation_length=args.continuation_length,
            num_prompts=args.num_prompts,
        )

        results_payload = json.loads(results_path.read_text(encoding="utf-8"))
        runs = choose_runs(results_payload, all_seeds=args.all_seeds)
        dataset_runs = []
        for run_payload in runs:
            checkpoint_path = results_path.parent / f"{sanitize_filename(run_payload['model_name'])}_seed_{run_payload['seed']}.pt"
            print(f"benchmarking dataset={dataset_id} model={run_payload['model_name']} seed={run_payload['seed']}", flush=True)
            dataset_runs.append(
                benchmark_run(
                    run_payload=run_payload,
                    dataset=dataset,
                    device=device,
                    checkpoint_path=checkpoint_path,
                    prompt_cases=prompt_cases,
                    sample_temperature=args.sample_temperature,
                )
            )

        payload["datasets"].append(
            {
                "dataset_id": dataset_id,
                "display_name": spec.display_name,
                "results_path": str(results_path),
                "runs": dataset_runs,
            }
        )

    json_path, md_path = write_outputs(args.experiment_dir, payload)
    print(f"generation_results_path={json_path}")
    print(f"generation_summary_path={md_path}")


if __name__ == "__main__":
    main()
