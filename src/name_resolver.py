"""
Indexed lexicon resolver for names and other high-value text fields.

This module is designed for the fixed LIC Form 300 pipeline where a large
local lexicon can be searched in-memory without external infrastructure.
It builds fast indices from local manifests and optional drop-in lexicon
files under ``data/lexicons``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PERSON_TOKEN_FIELDS = {
    "first_name",
    "middle_name",
    "last_name",
}

PERSON_FULL_FIELDS = {
    "father_full_name",
    "mother_full_name",
    "spouse_full_name",
    "nominee_name",
    "appointee_name",
    "declarant_name",
    "husband_full_name",
    "agent_name",
    "development_officer_name",
    "proposer_full_name",
    "proposer_father_name",
    "proposer_spouse_name",
}

EMPLOYER_FIELDS = {
    "employer_name",
    "branch_name",
}

PLAN_FIELDS = {
    "preferred_plan_name",
    "proposed_plan_name",
    "plan_name",
}

PERSON_TOKEN_CATEGORY = "person_token"
PERSON_FULL_CATEGORY = "person_full"
EMPLOYER_CATEGORY = "employer_name"
PLAN_CATEGORY = "plan_name"

ALL_CATEGORIES = (
    PERSON_TOKEN_CATEGORY,
    PERSON_FULL_CATEGORY,
    EMPLOYER_CATEGORY,
    PLAN_CATEGORY,
)

OCR_TO_ALPHA = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "2": "Z",
        "3": "E",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
        "9": "G",
    }
)

TOKEN_RE = re.compile(r"[A-Z0-9]+")
SPACE_RE = re.compile(r"\s+")
PERSON_STOP_TOKENS = {"MR", "MRS", "MS", "SHRI", "SMT", "KUMARI", "DR"}
UPPERCASE_KEEP = {"LIC", "HCL", "TCS", "WIPRO", "ONGC", "SBI", "PNB", "HDFC", "ICICI"}


@dataclass(frozen=True)
class LexiconEntry:
    text: str
    normalized: str
    category: str
    frequency: int
    tokens: Tuple[str, ...]
    token_codes: Tuple[str, ...]


@dataclass(frozen=True)
class ResolutionCandidate:
    text: str
    score: float
    category: str
    frequency: int
    exact: bool


@dataclass(frozen=True)
class ResolutionDecision:
    raw_text: str
    normalized_text: str
    resolved_text: str
    confidence: float
    category: str
    accepted: bool
    review_required: bool
    method: str
    reason: str
    candidates: Tuple[ResolutionCandidate, ...]


def _normalize(text: str, preserve_digits: bool = False) -> str:
    if not text:
        return ""
    clean = str(text).upper()
    if not preserve_digits:
        clean = clean.translate(OCR_TO_ALPHA)
    clean = re.sub(r"[^A-Z0-9\s/&\-.]", " ", clean)
    clean = clean.replace("/", " ")
    clean = clean.replace("-", " ")
    clean = clean.replace(".", " ")
    clean = SPACE_RE.sub(" ", clean).strip()
    return clean


def _tokenize(text: str) -> Tuple[str, ...]:
    return tuple(tok for tok in TOKEN_RE.findall(text) if tok)


def _ngrams(text: str, n: int = 3) -> Set[str]:
    collapsed = text.replace(" ", "")
    if not collapsed:
        return set()
    if len(collapsed) <= n:
        return {collapsed}
    return {collapsed[i:i + n] for i in range(len(collapsed) - n + 1)}


def _soundex(token: str) -> str:
    if not token:
        return ""
    mapping = {
        **{ch: "1" for ch in "BFPV"},
        **{ch: "2" for ch in "CGJKQSXZ"},
        **{ch: "3" for ch in "DT"},
        "L": "4",
        **{ch: "5" for ch in "MN"},
        "R": "6",
    }
    token = token.upper()
    first = token[0]
    encoded: List[str] = [first]
    last_digit = mapping.get(first, "")
    for ch in token[1:]:
        digit = mapping.get(ch, "")
        if digit != last_digit:
            if digit:
                encoded.append(digit)
            last_digit = digit
    return ("".join(encoded) + "000")[:4]


def _common_prefix_ratio(left: str, right: str, max_chars: int = 4) -> float:
    limit = min(len(left), len(right), max_chars)
    matched = 0
    for idx in range(limit):
        if left[idx] != right[idx]:
            break
        matched += 1
    return matched / max_chars if max_chars else 0.0


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / union if union else 0.0


def _smart_title(text: str) -> str:
    parts = []
    for part in SPACE_RE.sub(" ", text).strip().split(" "):
        if not part:
            continue
        upper = part.upper()
        if upper in UPPERCASE_KEEP or (len(upper) <= 3 and upper.isalpha() and upper == part):
            parts.append(upper)
        else:
            parts.append(upper.capitalize())
    return " ".join(parts)


def _field_to_categories(field_name: Optional[str]) -> Tuple[str, ...]:
    key = (field_name or "").strip().lower()
    if key in PERSON_TOKEN_FIELDS:
        return (PERSON_TOKEN_CATEGORY,)
    if key in PERSON_FULL_FIELDS:
        return (PERSON_FULL_CATEGORY, PERSON_TOKEN_CATEGORY)
    if key in EMPLOYER_FIELDS or "employer" in key or "branch" in key:
        return (EMPLOYER_CATEGORY,)
    if key in PLAN_FIELDS or "plan_name" in key:
        return (PLAN_CATEGORY,)
    if "name" in key or "father" in key or "mother" in key or "spouse" in key:
        return (PERSON_FULL_CATEGORY, PERSON_TOKEN_CATEGORY)
    return tuple()


def _score_candidate(query_norm: str, query_tokens: Tuple[str, ...], entry: LexiconEntry) -> float:
    char_score = SequenceMatcher(None, query_norm, entry.normalized).ratio()
    token_score = _jaccard(set(query_tokens), set(entry.tokens))
    ngram_score = _jaccard(_ngrams(query_norm), _ngrams(entry.normalized))
    phonetic_score = _jaccard(set(_soundex(tok) for tok in query_tokens), set(entry.token_codes))
    prefix_score = _common_prefix_ratio(query_norm, entry.normalized)
    freq_bonus = min(math.log1p(entry.frequency) / 20.0, 0.05)
    aligned_token_score = 0.0
    if query_tokens and entry.tokens and len(query_tokens) == len(entry.tokens):
        aligned_token_score = sum(
            SequenceMatcher(None, left, right).ratio()
            for left, right in zip(query_tokens, entry.tokens)
        ) / len(query_tokens)

    if len(query_tokens) == 1 and len(entry.tokens) == 1:
        token_score = max(token_score, char_score)
        aligned_token_score = max(aligned_token_score, char_score)

    score = (
        0.34 * char_score
        + 0.20 * token_score
        + 0.15 * ngram_score
        + 0.14 * aligned_token_score
        + 0.10 * phonetic_score
        + 0.07 * prefix_score
        + freq_bonus
    )
    if query_norm == entry.normalized:
        score += 0.15
    return min(score, 1.0)


class IndexedLexiconResolver:
    def __init__(
        self,
        manifest_paths: Optional[Sequence[Path]] = None,
        custom_lexicon_dir: Optional[Path] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        self.manifest_paths = tuple(
            manifest_paths
            if manifest_paths is not None
            else [
                repo_root / "data" / "form300_factory" / "manifests" / "synthetic_crops.jsonl",
                repo_root / "data" / "form300_factory" / "manifests" / "final_train_manifest.jsonl",
                repo_root / "data" / "form300_factory" / "manifests" / "final_val_manifest.jsonl",
            ]
        )
        self.custom_lexicon_dir = custom_lexicon_dir or (repo_root / "data" / "lexicons")

        self.entries: List[LexiconEntry] = []
        self.exact_index: DefaultDict[Tuple[str, str], Set[int]] = defaultdict(set)
        self.token_index: DefaultDict[Tuple[str, str], Set[int]] = defaultdict(set)
        self.token_code_index: DefaultDict[Tuple[str, str], Set[int]] = defaultdict(set)
        self.ngram_index: DefaultDict[Tuple[str, str], Set[int]] = defaultdict(set)
        self.length_index: DefaultDict[Tuple[str, int], Set[int]] = defaultdict(set)
        self.category_counts: Counter[str] = Counter()

        counters = self._load_counters()
        self._build_indices(counters)

    def resolve(
        self,
        raw_text: str,
        field_name: Optional[str] = None,
        top_k: int = 5,
    ) -> ResolutionDecision:
        categories = _field_to_categories(field_name)
        raw_text = (raw_text or "").strip()
        if not raw_text or not categories:
            return ResolutionDecision(
                raw_text=raw_text,
                normalized_text=_normalize(raw_text),
                resolved_text=_smart_title(_normalize(raw_text, preserve_digits=True)),
                confidence=0.0,
                category="",
                accepted=False,
                review_required=False,
                method="no_lexicon_route",
                reason="No matching lexicon category for field",
                candidates=tuple(),
            )

        query_norm = _normalize(raw_text)
        query_tokens = _tokenize(query_norm)
        if not query_norm or not query_tokens:
            return ResolutionDecision(
                raw_text=raw_text,
                normalized_text=query_norm,
                resolved_text=_smart_title(_normalize(raw_text, preserve_digits=True)),
                confidence=0.0,
                category=categories[0],
                accepted=False,
                review_required=False,
                method="empty_query",
                reason="Query was empty after normalization",
                candidates=tuple(),
            )

        candidate_ids: Set[int] = set()
        for category in categories:
            candidate_ids.update(self.exact_index.get((category, query_norm), set()))
            for token in query_tokens:
                candidate_ids.update(self.token_index.get((category, token), set()))
                candidate_ids.update(self.token_code_index.get((category, _soundex(token)), set()))
            for gram in _ngrams(query_norm):
                candidate_ids.update(self.ngram_index.get((category, gram), set()))
            for length in range(max(1, len(query_norm) - 3), len(query_norm) + 4):
                candidate_ids.update(self.length_index.get((category, length), set()))

        ranked: List[ResolutionCandidate] = []
        for idx in candidate_ids:
            entry = self.entries[idx]
            score = _score_candidate(query_norm, query_tokens, entry)
            if score < 0.55:
                continue
            ranked.append(
                ResolutionCandidate(
                    text=entry.text,
                    score=score,
                    category=entry.category,
                    frequency=entry.frequency,
                    exact=(entry.normalized == query_norm),
                )
            )

        deduped: Dict[Tuple[str, str], ResolutionCandidate] = {}
        for candidate in ranked:
            key = (candidate.category, candidate.text)
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = candidate
                continue
            if candidate.score > existing.score:
                deduped[key] = ResolutionCandidate(
                    text=candidate.text,
                    score=candidate.score,
                    category=candidate.category,
                    frequency=existing.frequency + candidate.frequency,
                    exact=(candidate.exact or existing.exact),
                )
            else:
                deduped[key] = ResolutionCandidate(
                    text=existing.text,
                    score=existing.score,
                    category=existing.category,
                    frequency=existing.frequency + candidate.frequency,
                    exact=(candidate.exact or existing.exact),
                )

        unique_ranked = sorted(
            deduped.values(),
            key=lambda item: (item.score, item.frequency, len(item.text)),
            reverse=True,
        )
        top_candidates = tuple(unique_ranked[:top_k])

        fallback_text = _smart_title(_normalize(raw_text, preserve_digits=True))
        if not top_candidates:
            return ResolutionDecision(
                raw_text=raw_text,
                normalized_text=query_norm,
                resolved_text=fallback_text,
                confidence=0.0,
                category=categories[0],
                accepted=False,
                review_required=False,
                method="no_candidate",
                reason="No lexicon candidates crossed the minimum score",
                candidates=tuple(),
            )

        best = top_candidates[0]
        second_score = top_candidates[1].score if len(top_candidates) > 1 else 0.0
        margin = best.score - second_score
        threshold = self._acceptance_threshold(best.category)
        accepted = best.exact or best.score >= (threshold + 0.03) or (
            best.score >= threshold and margin >= 0.05
        )
        review_required = not accepted and best.score >= (threshold - 0.07)

        if best.exact:
            method = "lexicon_exact"
            reason = "Exact normalized match found in local lexicon"
        elif accepted:
            method = "lexicon_fuzzy_accept"
            reason = f"Accepted fuzzy match with score={best.score:.3f} margin={margin:.3f}"
        elif review_required:
            method = "lexicon_fuzzy_review"
            reason = f"Candidate available but margin is too small score={best.score:.3f} margin={margin:.3f}"
        else:
            method = "lexicon_no_accept"
            reason = f"Best candidate below acceptance threshold score={best.score:.3f}"

        return ResolutionDecision(
            raw_text=raw_text,
            normalized_text=query_norm,
            resolved_text=best.text if accepted else fallback_text,
            confidence=best.score if accepted else min(best.score, 0.89),
            category=best.category,
            accepted=accepted,
            review_required=review_required,
            method=method,
            reason=reason,
            candidates=top_candidates,
        )

    def _acceptance_threshold(self, category: str) -> float:
        if category == PERSON_TOKEN_CATEGORY:
            return 0.88
        if category == PERSON_FULL_CATEGORY:
            return 0.92
        if category == EMPLOYER_CATEGORY:
            return 0.89
        if category == PLAN_CATEGORY:
            return 0.80
        return 0.90

    def _load_counters(self) -> Dict[str, Counter[str]]:
        counters: Dict[str, Counter[str]] = {category: Counter() for category in ALL_CATEGORIES}
        for path in self.manifest_paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    field_name = str(record.get("field_name", "")).strip().lower()
                    text = (
                        record.get("target_text")
                        or record.get("label_text")
                        or record.get("rendered_text")
                        or ""
                    )
                    self._ingest_text(counters, field_name, str(text))

        self._load_custom_lexicons(counters)
        return counters

    def _load_custom_lexicons(self, counters: Dict[str, Counter[str]]) -> None:
        if not self.custom_lexicon_dir.exists():
            return
        for path in self.custom_lexicon_dir.iterdir():
            if not path.is_file():
                continue
            category = path.stem.lower()
            if category not in counters:
                continue
            if path.suffix.lower() == ".txt":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        value = line.strip()
                        if value:
                            counters[category][_smart_title(value)] += 1
            elif path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        value = row.get("value") or row.get("text") or row.get("name")
                        if not value:
                            continue
                        count = int(row.get("count") or 1)
                        counters[category][_smart_title(value)] += max(1, count)
            elif path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, str):
                            counters[category][_smart_title(item)] += 1
                        elif isinstance(item, dict):
                            value = item.get("value") or item.get("text") or item.get("name")
                            if value:
                                counters[category][_smart_title(value)] += int(item.get("count") or 1)

    def _ingest_text(self, counters: Dict[str, Counter[str]], field_name: str, text: str) -> None:
        categories = _field_to_categories(field_name)
        if not categories or not text:
            return
        canonical = _smart_title(text)
        normalized = _normalize(canonical, preserve_digits=True)
        if not normalized:
            return

        if PERSON_FULL_CATEGORY in categories:
            counters[PERSON_FULL_CATEGORY][canonical] += 1
            for token in _tokenize(_normalize(canonical)):
                if token in PERSON_STOP_TOKENS or len(token) < 2:
                    continue
                counters[PERSON_TOKEN_CATEGORY][_smart_title(token)] += 1
            return

        for category in categories:
            counters[category][canonical] += 1

    def _build_indices(self, counters: Dict[str, Counter[str]]) -> None:
        for category, category_counter in counters.items():
            for text, frequency in category_counter.items():
                normalized = _normalize(text)
                tokens = _tokenize(normalized)
                if not normalized or not tokens:
                    continue
                entry = LexiconEntry(
                    text=_smart_title(text),
                    normalized=normalized,
                    category=category,
                    frequency=frequency,
                    tokens=tokens,
                    token_codes=tuple(_soundex(token) for token in tokens),
                )
                idx = len(self.entries)
                self.entries.append(entry)
                self.category_counts[category] += 1
                self.exact_index[(category, normalized)].add(idx)
                self.length_index[(category, len(normalized))].add(idx)
                for token in tokens:
                    self.token_index[(category, token)].add(idx)
                    self.token_code_index[(category, _soundex(token))].add(idx)
                for gram in _ngrams(normalized):
                    self.ngram_index[(category, gram)].add(idx)


@lru_cache(maxsize=1)
def get_default_name_resolver() -> IndexedLexiconResolver:
    return IndexedLexiconResolver()
