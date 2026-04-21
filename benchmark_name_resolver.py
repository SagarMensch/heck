"""
Benchmark the in-memory name resolver with large lexicon dictionaries.

This script can scale starter lexicons up to a target size such as 100,000
entries and then measure resolver load time and query latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.name_resolver import (
    ALL_CATEGORIES,
    EMPLOYER_CATEGORY,
    IndexedLexiconResolver,
    PERSON_FULL_CATEGORY,
    PERSON_TOKEN_CATEGORY,
    PLAN_CATEGORY,
)


CATEGORY_TO_FIELD = {
    PERSON_TOKEN_CATEGORY: "first_name",
    PERSON_FULL_CATEGORY: "father_full_name",
    EMPLOYER_CATEGORY: "employer_name",
    PLAN_CATEGORY: "preferred_plan_name",
}

TOKEN_SUFFIXES = [
    "Aster",
    "Nova",
    "Prime",
    "Vertex",
    "Sigma",
    "Orion",
    "Apex",
    "Zen",
    "Vista",
    "Nimbus",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark large lexicon lookup speed.")
    parser.add_argument("--target-size", type=int, default=100000, help="Target total number of generated lexicon entries.")
    parser.add_argument("--queries", type=int, default=5000, help="Number of benchmark queries to execute.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--lexicon-dir",
        type=Path,
        default=Path("data/lexicons"),
        help="Starter lexicon directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/lexicons/benchmark_latest.json"),
        help="Path to write JSON benchmark summary.",
    )
    parser.add_argument(
        "--keep-generated-dir",
        type=Path,
        default=None,
        help="Optional directory to keep the expanded benchmark lexicons.",
    )
    return parser.parse_args()


def _read_lexicon_csv(path: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not path.exists():
        return counter
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = (row.get("value") or "").strip()
            if not value:
                continue
            count = int(row.get("count") or 1)
            counter[value] += max(1, count)
    return counter


def _load_base_lexicons(lexicon_dir: Path) -> Dict[str, Counter[str]]:
    return {category: _read_lexicon_csv(lexicon_dir / f"{category}.csv") for category in ALL_CATEGORIES}


def _mutate_text(text: str, rng: random.Random) -> str:
    replacements = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})
    if not text:
        return text
    mutated = text.translate(replacements)
    if len(mutated) > 4 and rng.random() < 0.5:
        pos = rng.randrange(1, len(mutated) - 1)
        mutated = mutated[:pos] + mutated[pos + 1 :]
    elif len(mutated) > 3 and rng.random() < 0.5:
        pos = rng.randrange(1, len(mutated))
        mutated = mutated[:pos] + mutated[pos - 1] + mutated[pos:]
    return mutated


def _sized_value(base: str, category: str, idx: int) -> str:
    suffix = TOKEN_SUFFIXES[idx % len(TOKEN_SUFFIXES)]
    serial = idx // len(TOKEN_SUFFIXES)
    if category == PERSON_TOKEN_CATEGORY:
        return f"{base} {suffix}{serial}"
    if category == PERSON_FULL_CATEGORY:
        return f"{base} {suffix} {serial}"
    if category == EMPLOYER_CATEGORY:
        return f"{base} {suffix} Group {serial}"
    if category == PLAN_CATEGORY:
        return f"{base} {suffix} Variant {serial}"
    return f"{base} {suffix} {serial}"


def _expand_lexicons(base_lexicons: Dict[str, Counter[str]], target_size: int) -> Dict[str, Counter[str]]:
    expanded: Dict[str, Counter[str]] = {category: Counter(counter) for category, counter in base_lexicons.items()}
    base_total = sum(len(counter) for counter in expanded.values())
    if base_total >= target_size:
        return expanded

    weights = {
        PERSON_TOKEN_CATEGORY: 0.45,
        PERSON_FULL_CATEGORY: 0.35,
        EMPLOYER_CATEGORY: 0.10,
        PLAN_CATEGORY: 0.10,
    }
    additions_needed = target_size - base_total

    for category, weight in weights.items():
        desired = int(additions_needed * weight)
        counter = expanded[category]
        seeds = list(counter.keys()) or [f"Seed {category}"]
        idx = 0
        while desired > 0:
            seed = seeds[idx % len(seeds)]
            value = _sized_value(seed, category, idx)
            if value not in counter:
                counter[value] = 1
                desired -= 1
            idx += 1

    current_total = sum(len(counter) for counter in expanded.values())
    filler_idx = 0
    while current_total < target_size:
        value = f"Fallback Person {filler_idx}"
        if value not in expanded[PERSON_FULL_CATEGORY]:
            expanded[PERSON_FULL_CATEGORY][value] = 1
            current_total += 1
        filler_idx += 1
    return expanded


def _write_expanded_lexicons(lexicons: Dict[str, Counter[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for category, counter in lexicons.items():
        with (output_dir / f"{category}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["value", "count"])
            for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
                writer.writerow([value, count])


def _build_queries(lexicons: Dict[str, Counter[str]], query_count: int, rng: random.Random) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    category_pool: List[Tuple[str, Sequence[str]]] = [
        (category, list(counter.keys()))
        for category, counter in lexicons.items()
        if counter
    ]
    while len(triples) < query_count and category_pool:
        category, values = rng.choice(category_pool)
        value = rng.choice(values)
        raw_query = value if rng.random() < 0.4 else _mutate_text(value, rng)
        triples.append((raw_query, CATEGORY_TO_FIELD[category], category))
    return triples


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    repo_root = Path(__file__).resolve().parent
    lexicon_dir = (repo_root / args.lexicon_dir).resolve()
    output_json = (repo_root / args.output_json).resolve()

    base_lexicons = _load_base_lexicons(lexicon_dir)
    expanded = _expand_lexicons(base_lexicons, args.target_size)

    if args.keep_generated_dir:
        generated_dir = (repo_root / args.keep_generated_dir).resolve()
        _write_expanded_lexicons(expanded, generated_dir)
    else:
        tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="lexicon_bench_", dir=str((repo_root / "data").resolve()))
        generated_dir = Path(tmp_dir_ctx.name)
        _write_expanded_lexicons(expanded, generated_dir)

    queries = _build_queries(expanded, args.queries, rng)

    load_start = time.perf_counter()
    resolver = IndexedLexiconResolver(manifest_paths=(), custom_lexicon_dir=generated_dir)
    load_seconds = time.perf_counter() - load_start

    latencies_ms: List[float] = []
    accepted = 0
    review_required = 0
    for raw_query, field_name, _category in queries:
        t0 = time.perf_counter()
        decision = resolver.resolve(raw_query, field_name=field_name)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        accepted += int(decision.accepted)
        review_required += int(decision.review_required)

    latencies_ms.sort()
    summary = {
        "target_size": args.target_size,
        "actual_total_entries": int(sum(resolver.category_counts.values())),
        "category_counts": {key: int(value) for key, value in resolver.category_counts.items()},
        "queries": len(queries),
        "load_seconds": round(load_seconds, 6),
        "mean_query_ms": round(statistics.mean(latencies_ms), 6) if latencies_ms else 0.0,
        "median_query_ms": round(_percentile(latencies_ms, 0.50), 6),
        "p95_query_ms": round(_percentile(latencies_ms, 0.95), 6),
        "p99_query_ms": round(_percentile(latencies_ms, 0.99), 6),
        "accepted_rate": round(accepted / max(len(queries), 1), 6),
        "review_rate": round(review_required / max(len(queries), 1), 6),
        "generated_lexicon_dir": str(generated_dir),
        "source_lexicon_dir": str(lexicon_dir),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not args.keep_generated_dir:
        tmp_dir_ctx.cleanup()


if __name__ == "__main__":
    main()
