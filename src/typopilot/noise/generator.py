"""Severity-profiled, reproducible noise generation.

The generator turns one clean sentence into one corrupted sentence by
applying a sampled sequence of operations from :mod:`typopilot.noise.ops`.

Reproducibility contract: ``corrupt(text, severity, example_id, seed)`` is a
pure function of its arguments. The per-example RNG is derived from
``(seed, example_id, severity)``, so corrupting a dataset is stable under
re-ordering, subsetting, and parallelisation.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .ops import OPS

# How many times to re-draw an operation that returned None (not applicable)
# before giving up on that slot.
_MAX_DRAWS_PER_SLOT = 8


@dataclass(frozen=True)
class SeverityProfile:
    """How aggressively to corrupt, and with which operation mix."""

    name: str
    #: Inclusive range for the number of operations to attempt.
    ops_range: tuple[int, int]
    #: Recoverability cap: at most this many operations per input word.
    #: Keeps short inputs from being corrupted beyond recognition.
    max_ops_per_word: float
    #: Operation name -> sampling weight. Names must exist in ``OPS``.
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Op-name validation happens in NoiseGenerator, which knows the full
        # op table (stateless OPS plus dictionary-backed ops like
        # real_misspelling that only exist when a dictionary is supplied).
        if not self.weights:
            raise ValueError(f"profile {self.name!r} has no operation weights")


def _default_weights(**overrides: float) -> dict[str, float]:
    weights = {
        "swap_adjacent_chars": 3.0,
        "delete_char": 2.5,
        "duplicate_char": 1.5,
        "insert_random_char": 1.0,
        "nearby_key_substitution": 3.0,
        "split_word": 2.0,
        "merge_words": 2.0,
        "add_extra_space": 1.0,
        "flip_case": 1.5,
        "lowercase_sentence_start": 2.0,
        "random_capitalize_word": 1.0,
        "remove_punctuation": 2.0,
        "duplicate_punctuation": 0.5,
        "replace_punctuation": 0.5,
        "delete_filler_word": 1.0,
        "repeat_word": 1.0,
        "swap_adjacent_words": 0.75,
        "abbreviate_word": 1.5,
        "drop_apostrophe": 1.5,
        # Real human misspellings (received -> recieved). Weighted high:
        # they are the dominant error class in real typing. Silently skipped
        # when the generator has no misspelling dictionary.
        "real_misspelling": 3.0,
    }
    weights.update(overrides)
    return weights


#: The four severity buckets from the spec (idea.md section 7).
PROFILES: dict[str, SeverityProfile] = {
    "light": SeverityProfile(
        name="light",
        ops_range=(1, 2),
        max_ops_per_word=0.35,
        weights=_default_weights(),
    ),
    "medium": SeverityProfile(
        name="medium",
        ops_range=(2, 4),
        max_ops_per_word=0.6,
        weights=_default_weights(),
    ),
    "heavy": SeverityProfile(
        name="heavy",
        ops_range=(4, 7),
        max_ops_per_word=0.9,
        weights=_default_weights(),
    ),
    "extreme": SeverityProfile(
        name="extreme",
        ops_range=(7, 12),
        max_ops_per_word=1.2,
        # Extreme leans harder into keyboard-style damage but drops the
        # meaning-affecting word ops, so the input stays recoverable.
        weights=_default_weights(
            nearby_key_substitution=4.0,
            swap_adjacent_chars=4.0,
            merge_words=3.0,
            split_word=3.0,
            delete_filler_word=0.5,
            swap_adjacent_words=0.25,
        ),
    ),
}


@dataclass(frozen=True)
class CorruptionResult:
    clean: str
    noisy: str
    severity: str
    #: One log entry per applied operation, in application order.
    ops_log: tuple[dict, ...]
    #: The derived per-example seed actually used (for exact replay).
    example_seed: int

    def to_record(self, example_id: str | None = None) -> dict:
        """JSONL-ready training-pair record."""
        record = {
            "input": self.noisy,
            "target": self.clean,
            "severity": self.severity,
            "ops": list(self.ops_log),
            "example_seed": self.example_seed,
        }
        if example_id is not None:
            record["id"] = example_id
        return record


def _derive_seed(seed: int, example_id: str, severity: str) -> int:
    payload = f"{seed}:{example_id}:{severity}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class NoiseGenerator:
    """Applies a severity profile to clean text, deterministically."""

    def __init__(
        self,
        seed: int = 13,
        profiles: dict[str, SeverityProfile] | None = None,
        misspellings: dict[str, list[str]] | None = None,
    ):
        self.seed = seed
        self.profiles = profiles or PROFILES
        self.ops = dict(OPS)
        if misspellings:
            from .ops import make_misspelling_op
            self.ops["real_misspelling"] = make_misspelling_op(misspellings)

        # Validate profiles against the ops actually available; weights for
        # dictionary-backed ops that were not supplied are dropped silently
        # so the same profile definitions work with or without a dictionary.
        for profile in self.profiles.values():
            known = {n: w for n, w in profile.weights.items() if n in self.ops}
            unknown = set(profile.weights) - set(OPS) - {"real_misspelling"}
            if unknown:
                raise ValueError(
                    f"profile {profile.name!r} references unknown ops: {sorted(unknown)}"
                )
            object.__setattr__(profile, "weights", known)

    def corrupt(
        self,
        text: str,
        severity: str = "medium",
        example_id: str = "0",
    ) -> CorruptionResult:
        if severity not in self.profiles:
            raise KeyError(
                f"unknown severity {severity!r}; expected one of {sorted(self.profiles)}"
            )
        profile = self.profiles[severity]
        example_seed = _derive_seed(self.seed, example_id, severity)
        rng = random.Random(example_seed)

        word_count = max(1, len(text.split()))
        n_ops = rng.randint(*profile.ops_range)
        n_ops = min(n_ops, max(1, round(word_count * profile.max_ops_per_word)))

        op_names = list(profile.weights)
        op_weights = [profile.weights[name] for name in op_names]

        current = text
        ops_log: list[dict] = []
        for _ in range(n_ops):
            for _ in range(_MAX_DRAWS_PER_SLOT):
                name = rng.choices(op_names, weights=op_weights, k=1)[0]
                result = self.ops[name](current, rng)
                if result is not None:
                    current, entry = result
                    ops_log.append(entry)
                    break

        return CorruptionResult(
            clean=text,
            noisy=current,
            severity=severity,
            ops_log=tuple(ops_log),
            example_seed=example_seed,
        )

    def corrupt_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        severity_mix: dict[str, float] | None = None,
        variants_per_sentence: int = 1,
        text_key: str = "text",
    ) -> dict:
        """Corrupt a JSONL file of clean sentences into training pairs.

        ``severity_mix`` maps severity name -> proportion (defaults to an
        even mix of the four buckets). Each input sentence produces
        ``variants_per_sentence`` corrupted variants. Streams line by line;
        never loads the whole file into memory.

        Returns summary statistics for the dataset report.
        """
        severity_mix = severity_mix or {s: 1.0 for s in self.profiles}
        mix_names = list(severity_mix)
        mix_weights = [severity_mix[name] for name in mix_names]

        input_path, output_path = Path(input_path), Path(output_path)
        stats: dict = {
            "input_sentences": 0,
            "pairs_written": 0,
            "unchanged_pairs": 0,
            "severity_counts": {name: 0 for name in mix_names},
            "op_counts": {},
        }

        with input_path.open(encoding="utf-8") as src, \
                output_path.open("w", encoding="utf-8") as dst:
            for line_no, line in enumerate(src):
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line)[text_key]
                stats["input_sentences"] += 1
                for variant in range(variants_per_sentence):
                    example_id = f"{line_no}:{variant}"
                    severity_rng = random.Random(
                        _derive_seed(self.seed, example_id, "severity-choice")
                    )
                    severity = severity_rng.choices(mix_names, weights=mix_weights, k=1)[0]
                    result = self.corrupt(text, severity, example_id=example_id)

                    stats["pairs_written"] += 1
                    stats["severity_counts"][severity] += 1
                    if result.noisy == result.clean:
                        stats["unchanged_pairs"] += 1
                    for entry in result.ops_log:
                        op = entry["op"]
                        stats["op_counts"][op] = stats["op_counts"].get(op, 0) + 1

                    dst.write(
                        json.dumps(result.to_record(example_id), ensure_ascii=False) + "\n"
                    )
        return stats
