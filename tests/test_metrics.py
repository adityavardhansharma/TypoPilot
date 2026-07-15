from typopilot.eval.metrics import (
    aggregate,
    cer,
    cer_multi,
    correction_rates,
    exact_match,
    exact_match_multi,
    levenshtein,
    wer,
    wer_multi,
)


def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein(["a", "b"], ["a", "c"]) == 1


def test_cer():
    assert cer("what", "what") == 0.0
    assert cer("waht", "what") == 0.5  # one transposition = 2 char edits / 4
    assert cer("", "") == 0.0
    assert cer("x", "") == 1.0


def test_wer():
    assert wer("the task", "the task") == 0.0
    assert wer("the the task", "the task") == 0.5
    assert wer("", "the task") == 1.0


def test_exact_match():
    assert exact_match("a b", "a b")
    assert not exact_match("a b", "a  b")  # whitespace counts


def test_multi_reference_takes_best():
    refs = ["What is the weather?", "What's the weather?"]
    assert exact_match_multi("What's the weather?", refs)
    assert cer_multi("What's the weather?", refs) == 0.0
    assert wer_multi("What's the weather?", refs) == 0.0


def test_correction_rates_improvement():
    rates = correction_rates("waht is it", "what is it", "what is it")
    assert rates["output_cer"] == 0.0
    assert rates["improvement"] > 0
    assert not rates["overcorrected"]
    assert not rates["undercorrected"]


def test_correction_rates_overcorrection():
    # Output strays further from the reference than the input was.
    rates = correction_rates("what is it", "completely different text", "what is it")
    assert rates["overcorrected"]


def test_aggregate():
    rows = [{"cer": 0.5, "em": True}, {"cer": 0.0, "em": False}]
    agg = aggregate(rows)
    assert agg == {"cer": 0.25, "em": 0.5, "n": 2}
