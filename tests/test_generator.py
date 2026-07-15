import json

import pytest

from typopilot.noise.generator import PROFILES, NoiseGenerator

SAMPLE = "What is the weather, and how did the AI perform in the task given to it?"


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def test_same_seed_and_id_reproduces_exactly():
    a = NoiseGenerator(seed=13).corrupt(SAMPLE, "heavy", example_id="x")
    b = NoiseGenerator(seed=13).corrupt(SAMPLE, "heavy", example_id="x")
    assert a == b


def test_different_example_ids_differ():
    gen = NoiseGenerator(seed=13)
    outputs = {gen.corrupt(SAMPLE, "heavy", example_id=str(i)).noisy for i in range(10)}
    assert len(outputs) > 1


def test_different_seeds_differ():
    a = NoiseGenerator(seed=1).corrupt(SAMPLE, "heavy", example_id="x")
    b = NoiseGenerator(seed=2).corrupt(SAMPLE, "heavy", example_id="x")
    assert a.noisy != b.noisy


def test_severity_increases_average_damage():
    gen = NoiseGenerator(seed=13)
    means = {}
    for severity in ("light", "medium", "heavy", "extreme"):
        distances = [
            levenshtein(SAMPLE, gen.corrupt(SAMPLE, severity, example_id=str(i)).noisy)
            for i in range(40)
        ]
        means[severity] = sum(distances) / len(distances)
    assert means["light"] < means["medium"] < means["heavy"] < means["extreme"]


def test_output_never_empty_and_ops_logged():
    gen = NoiseGenerator(seed=7)
    for severity in PROFILES:
        for i in range(20):
            result = gen.corrupt(SAMPLE, severity, example_id=str(i))
            assert result.noisy.strip()
            assert result.ops_log, "every corruption must log its operations"
            assert all("op" in entry for entry in result.ops_log)


def test_short_input_is_capped_not_destroyed():
    gen = NoiseGenerator(seed=13)
    short = "ok thanks"
    for i in range(20):
        result = gen.corrupt(short, "extreme", example_id=str(i))
        # recoverability cap: 2 words * 1.2 ops/word -> at most 2 ops applied
        assert len(result.ops_log) <= 2
        assert result.noisy.strip()


def test_unknown_severity_raises():
    with pytest.raises(KeyError):
        NoiseGenerator().corrupt(SAMPLE, "apocalyptic")


def test_corrupt_file_round_trip(tmp_path):
    sentences = [
        "What is the weather today?",
        "The model performed well on the task.",
        "Please send me the report before the meeting.",
    ]
    src = tmp_path / "clean.jsonl"
    dst = tmp_path / "pairs.jsonl"
    src.write_text(
        "\n".join(json.dumps({"text": s}) for s in sentences), encoding="utf-8"
    )

    gen = NoiseGenerator(seed=13)
    stats = gen.corrupt_file(src, dst, variants_per_sentence=2)

    records = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    assert stats["input_sentences"] == 3
    assert stats["pairs_written"] == 6
    assert len(records) == 6
    for record in records:
        assert record["target"] in sentences
        assert record["input"].strip()
        assert record["ops"]
        assert record["severity"] in PROFILES
    assert sum(stats["severity_counts"].values()) == 6
    assert sum(stats["op_counts"].values()) > 0


def test_corrupt_file_is_reproducible(tmp_path):
    src = tmp_path / "clean.jsonl"
    src.write_text(json.dumps({"text": SAMPLE}), encoding="utf-8")

    out_a, out_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    NoiseGenerator(seed=13).corrupt_file(src, out_a, variants_per_sentence=3)
    NoiseGenerator(seed=13).corrupt_file(src, out_b, variants_per_sentence=3)
    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")
