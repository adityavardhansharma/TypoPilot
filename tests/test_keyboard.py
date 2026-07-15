import string

from typopilot.noise.keyboard import QWERTY_ADJACENCY, nearby_keys


def test_g_neighbours_match_spec_example():
    # idea.md section 8: g -> f, h, t, y, v, b
    assert set(QWERTY_ADJACENCY["g"]) == set("fhtyvb")


def test_all_letters_present_with_neighbours():
    for letter in string.ascii_lowercase:
        assert QWERTY_ADJACENCY[letter], f"{letter} has no neighbours"


def test_adjacency_is_symmetric():
    for key, neighbours in QWERTY_ADJACENCY.items():
        for other in neighbours:
            assert key in QWERTY_ADJACENCY[other], f"{key}->{other} not symmetric"


def test_no_key_is_its_own_neighbour():
    for key, neighbours in QWERTY_ADJACENCY.items():
        assert key not in neighbours


def test_nearby_keys_preserves_case():
    assert nearby_keys("G") == nearby_keys("g").upper()
    assert nearby_keys("g").islower()


def test_nearby_keys_empty_for_unknown_chars():
    for ch in (" ", ".", "é", "ह"):
        assert nearby_keys(ch) == ""


def test_corner_keys_have_few_neighbours():
    # q sits in a corner of the letter grid: w and a beside it, 1 and 2 above.
    # (s is excluded by the row stagger — it is > 1.3 key-widths from q.)
    assert set(QWERTY_ADJACENCY["q"]) == set("wa12")
