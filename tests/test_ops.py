import random

import pytest

from typopilot.noise.keyboard import QWERTY_ADJACENCY
from typopilot.noise.ops import OPS

SAMPLE = "What is the weather, and how did the AI perform in the task given to it?"

# Ops that need specific material in the sentence to fire.
_OP_SAMPLES = {
    "abbreviate_word": "Can you please tell me about the meeting before tomorrow?",
    "drop_apostrophe": "I don't think it's the model's fault.",
}


@pytest.mark.parametrize("name", sorted(OPS))
def test_op_changes_text_and_logs(name):
    op = OPS[name]
    sample = _OP_SAMPLES.get(name, SAMPLE)
    result = op(sample, random.Random(0))
    assert result is not None, f"{name} not applicable to a rich sample sentence"
    corrupted, entry = result
    assert corrupted != sample
    assert entry["op"] == name


@pytest.mark.parametrize("name", sorted(OPS))
def test_op_is_deterministic_given_seed(name):
    op = OPS[name]
    a = op(SAMPLE, random.Random(42))
    b = op(SAMPLE, random.Random(42))
    assert a == b


@pytest.mark.parametrize("name", sorted(OPS))
def test_op_returns_none_on_empty_text(name):
    assert OPS[name]("", random.Random(0)) is None


def test_nearby_key_substitution_uses_adjacency():
    for seed in range(30):
        corrupted, entry = OPS["nearby_key_substitution"](SAMPLE, random.Random(seed))
        original, replacement = entry["char"], entry["replacement"]
        assert replacement.lower() in QWERTY_ADJACENCY[original.lower()]


def test_swap_adjacent_chars_only_reorders():
    corrupted, _ = OPS["swap_adjacent_chars"](SAMPLE, random.Random(1))
    assert sorted(corrupted) == sorted(SAMPLE)


def test_delete_char_refuses_very_short_text():
    assert OPS["delete_char"]("hi", random.Random(0)) is None


def test_merge_words_removes_one_space():
    corrupted, _ = OPS["merge_words"]("the task given", random.Random(0))
    assert corrupted.count(" ") == 1
    assert corrupted.replace(" ", "") == "thetaskgiven"


def test_split_word_keeps_characters():
    corrupted, entry = OPS["split_word"]("perform", random.Random(0))
    assert corrupted.replace(" ", "") == "perform"
    assert " " in corrupted


def test_delete_filler_word_never_touches_content_words():
    for seed in range(30):
        result = OPS["delete_filler_word"](SAMPLE, random.Random(seed))
        assert result is not None
        _, entry = result
        assert entry["word"].lower().strip("?,.") in {
            "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
            "to", "of", "in", "on", "at", "it", "so", "do", "did", "had",
            "has", "and", "or", "but",
        }


def test_abbreviate_word_preserves_trailing_punctuation():
    corrupted, entry = OPS["abbreviate_word"]("can you come, please?", random.Random(0))
    # whichever word was abbreviated, punctuation must survive
    assert corrupted.endswith("?")
    assert "," in corrupted


def test_lowercase_sentence_start():
    corrupted, _ = OPS["lowercase_sentence_start"]("What is this", random.Random(0))
    assert corrupted == "what is this"
    # not applicable when the sentence already starts lowercase
    assert OPS["lowercase_sentence_start"]("what is this", random.Random(0)) is None
