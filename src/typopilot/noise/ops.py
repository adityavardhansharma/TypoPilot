"""Individual corruption operations.

Every operation has the signature::

    op(text: str, rng: random.Random) -> tuple[str, dict] | None

It returns the corrupted text plus a log entry describing exactly what was
done, or ``None`` when the operation is not applicable to this text (too
short, no eligible characters, ...). Returning ``None`` lets the generator
pick a different operation instead of silently emitting an unchanged pair.

All randomness must come from the supplied ``rng`` so that corruption is
fully reproducible from a seed.
"""

from __future__ import annotations

import random
import string

from .keyboard import nearby_keys

# Words this short are candidates for accidental deletion ("I did not go"
# -> "I did not"), which must never happen to content-bearing words.
_DELETABLE_WORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "it", "so", "do", "did", "had", "has",
    "and", "or", "but",
}

# Common informal texting abbreviations (word-level errors, spec section 7).
_ABBREVIATIONS = {
    "you": "u",
    "your": "ur",
    "are": "r",
    "please": "pls",
    "people": "ppl",
    "thanks": "thnx",
    "because": "bcoz",
    "message": "msg",
    "tomorrow": "tmrw",
    "great": "gr8",
    "before": "b4",
    "see": "c",
    "okay": "ok",
    "really": "rly",
    "probably": "prolly",
    "about": "abt",
    "right": "rite",
    "should": "shud",
    "would": "wud",
    "know": "kno",
}

_PUNCT_REPLACEMENTS = {
    ".": ",;",
    ",": ".;",
    "?": ".!",
    "!": ".?",
    ";": ",:",
    ":": ";,",
    "'": "",
}

_INSERTABLE_PUNCT = ".,?!;"


def _letter_positions(text: str) -> list[int]:
    return [i for i, ch in enumerate(text) if ch.isalpha()]


def _words(text: str) -> list[str]:
    return text.split()


# ---------------------------------------------------------------------------
# Character-level operations
# ---------------------------------------------------------------------------

def swap_adjacent_chars(text: str, rng: random.Random):
    """what -> waht"""
    positions = [
        i for i in _letter_positions(text)[:-1]
        if text[i + 1].isalpha() and text[i] != text[i + 1]
    ]
    if not positions:
        return None
    i = rng.choice(positions)
    out = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return out, {"op": "swap_adjacent_chars", "pos": i, "chars": text[i:i + 2]}


def delete_char(text: str, rng: random.Random):
    """weather -> wether"""
    positions = _letter_positions(text)
    if len(positions) < 4:  # keep very short texts recoverable
        return None
    i = rng.choice(positions)
    out = text[:i] + text[i + 1:]
    return out, {"op": "delete_char", "pos": i, "char": text[i]}


def duplicate_char(text: str, rng: random.Random):
    """the -> thhe"""
    positions = _letter_positions(text)
    if not positions:
        return None
    i = rng.choice(positions)
    out = text[:i] + text[i] + text[i:]
    return out, {"op": "duplicate_char", "pos": i, "char": text[i]}


def insert_random_char(text: str, rng: random.Random):
    """task -> taksk"""
    positions = _letter_positions(text)
    if not positions:
        return None
    i = rng.choice(positions)
    ch = rng.choice(string.ascii_lowercase)
    out = text[:i] + ch + text[i:]
    return out, {"op": "insert_random_char", "pos": i, "char": ch}


def nearby_key_substitution(text: str, rng: random.Random):
    """given -> giben (uses real QWERTY adjacency)"""
    positions = [i for i in _letter_positions(text) if nearby_keys(text[i])]
    if not positions:
        return None
    i = rng.choice(positions)
    replacement = rng.choice(nearby_keys(text[i]))
    out = text[:i] + replacement + text[i + 1:]
    return out, {
        "op": "nearby_key_substitution",
        "pos": i,
        "char": text[i],
        "replacement": replacement,
    }


# ---------------------------------------------------------------------------
# Word-boundary operations
# ---------------------------------------------------------------------------

def split_word(text: str, rng: random.Random):
    """perform -> perfo rm"""
    words = _words(text)
    candidates = [i for i, w in enumerate(words) if sum(c.isalpha() for c in w) >= 4]
    if not candidates:
        return None
    wi = rng.choice(candidates)
    word = words[wi]
    split_at = rng.randint(2, len(word) - 2)
    words[wi] = word[:split_at] + " " + word[split_at:]
    return " ".join(words), {"op": "split_word", "word": word, "split_at": split_at}


def merge_words(text: str, rng: random.Random):
    """the task -> thetask"""
    words = _words(text)
    if len(words) < 2:
        return None
    wi = rng.randint(0, len(words) - 2)
    merged = words[wi] + words[wi + 1]
    out_words = words[:wi] + [merged] + words[wi + 2:]
    return " ".join(out_words), {"op": "merge_words", "words": words[wi:wi + 2]}


def add_extra_space(text: str, rng: random.Random):
    """the task -> the  task"""
    space_positions = [i for i, ch in enumerate(text) if ch == " "]
    if not space_positions:
        return None
    i = rng.choice(space_positions)
    out = text[:i] + " " + text[i:]
    return out, {"op": "add_extra_space", "pos": i}


# ---------------------------------------------------------------------------
# Capitalisation operations
# ---------------------------------------------------------------------------

def flip_case(text: str, rng: random.Random):
    """What -> what / the -> The, applied to one letter"""
    positions = _letter_positions(text)
    if not positions:
        return None
    i = rng.choice(positions)
    flipped = text[i].lower() if text[i].isupper() else text[i].upper()
    out = text[:i] + flipped + text[i + 1:]
    return out, {"op": "flip_case", "pos": i, "char": text[i]}


def lowercase_sentence_start(text: str, rng: random.Random):
    """What is -> what is"""
    for i, ch in enumerate(text):
        if ch.isalpha():
            if not ch.isupper():
                return None
            out = text[:i] + ch.lower() + text[i + 1:]
            return out, {"op": "lowercase_sentence_start", "pos": i}
    return None


def random_capitalize_word(text: str, rng: random.Random):
    """is the -> Is ThE (random inner capitalisation)"""
    words = _words(text)
    candidates = [i for i, w in enumerate(words) if any(c.islower() for c in w)]
    if not candidates:
        return None
    wi = rng.choice(candidates)
    word = words[wi]
    letter_idx = rng.choice([i for i, c in enumerate(word) if c.islower()])
    words[wi] = word[:letter_idx] + word[letter_idx].upper() + word[letter_idx + 1:]
    return " ".join(words), {"op": "random_capitalize_word", "word": word}


# ---------------------------------------------------------------------------
# Punctuation operations
# ---------------------------------------------------------------------------

def remove_punctuation(text: str, rng: random.Random):
    positions = [i for i, ch in enumerate(text) if ch in _PUNCT_REPLACEMENTS]
    if not positions:
        return None
    i = rng.choice(positions)
    out = (text[:i] + text[i + 1:]).replace("  ", " ")
    return out, {"op": "remove_punctuation", "pos": i, "char": text[i]}


def duplicate_punctuation(text: str, rng: random.Random):
    positions = [i for i, ch in enumerate(text) if ch in _INSERTABLE_PUNCT]
    if not positions:
        return None
    i = rng.choice(positions)
    out = text[:i] + text[i] + text[i:]
    return out, {"op": "duplicate_punctuation", "pos": i, "char": text[i]}


def replace_punctuation(text: str, rng: random.Random):
    positions = [
        i for i, ch in enumerate(text)
        if ch in _PUNCT_REPLACEMENTS and _PUNCT_REPLACEMENTS[ch]
    ]
    if not positions:
        return None
    i = rng.choice(positions)
    replacement = rng.choice(_PUNCT_REPLACEMENTS[text[i]])
    out = text[:i] + replacement + text[i + 1:]
    return out, {
        "op": "replace_punctuation",
        "pos": i,
        "char": text[i],
        "replacement": replacement,
    }


# ---------------------------------------------------------------------------
# Word-level operations
# ---------------------------------------------------------------------------

def delete_filler_word(text: str, rng: random.Random):
    """I am going to the shop -> I am going to shop (only safe fillers)"""
    words = _words(text)
    if len(words) < 4:
        return None
    candidates = [
        i for i, w in enumerate(words)
        if w.lower().strip(string.punctuation) in _DELETABLE_WORDS
    ]
    if not candidates:
        return None
    wi = rng.choice(candidates)
    removed = words.pop(wi)
    return " ".join(words), {"op": "delete_filler_word", "word": removed}


def repeat_word(text: str, rng: random.Random):
    """the task -> the the task"""
    words = _words(text)
    if not words:
        return None
    wi = rng.randint(0, len(words) - 1)
    words.insert(wi, words[wi])
    return " ".join(words), {"op": "repeat_word", "word": words[wi]}


def swap_adjacent_words(text: str, rng: random.Random):
    """did the ai -> did ai the"""
    words = _words(text)
    if len(words) < 3:
        return None
    wi = rng.randint(0, len(words) - 2)
    if words[wi] == words[wi + 1]:
        return None
    words[wi], words[wi + 1] = words[wi + 1], words[wi]
    return " ".join(words), {"op": "swap_adjacent_words", "words": words[wi:wi + 2]}


def abbreviate_word(text: str, rng: random.Random):
    """you -> u, please -> pls"""
    words = _words(text)
    candidates = [
        i for i, w in enumerate(words)
        if w.lower().strip(string.punctuation) in _ABBREVIATIONS
    ]
    if not candidates:
        return None
    wi = rng.choice(candidates)
    word = words[wi]
    stripped = word.lower().strip(string.punctuation)
    trailing = word[len(word.rstrip(string.punctuation)):]
    words[wi] = _ABBREVIATIONS[stripped] + trailing
    return " ".join(words), {"op": "abbreviate_word", "word": word, "replacement": words[wi]}


def drop_apostrophe(text: str, rng: random.Random):
    """don't -> dont, I'm -> Im"""
    positions = [
        i for i, ch in enumerate(text)
        if ch == "'" and 0 < i < len(text) - 1
        and text[i - 1].isalpha() and text[i + 1].isalpha()
    ]
    if not positions:
        return None
    i = rng.choice(positions)
    out = text[:i] + text[i + 1:]
    return out, {"op": "drop_apostrophe", "pos": i}


def make_misspelling_op(mapping: dict[str, list[str]]):
    """Build a ``real_misspelling`` op from a {correct: [wrong, ...]} dict.

    Replaces a correctly spelled word with a real human misspelling
    (received -> recieved), preserving leading capitalisation and trailing
    punctuation. Returned function has the standard op signature.
    """

    def real_misspelling(text: str, rng: random.Random):
        words = _words(text)
        candidates = [
            i for i, w in enumerate(words)
            if w.lower().strip(string.punctuation) in mapping
        ]
        if not candidates:
            return None
        wi = rng.choice(candidates)
        word = words[wi]
        core = word.strip(string.punctuation)
        start = word.find(core)
        leading, trailing = word[:start], word[start + len(core):]
        wrong = rng.choice(mapping[core.lower()])
        if core[:1].isupper():
            wrong = wrong[:1].upper() + wrong[1:]
        words[wi] = leading + wrong + trailing
        return " ".join(words), {
            "op": "real_misspelling", "word": word, "replacement": words[wi],
        }

    return real_misspelling


#: Registry of every operation, keyed by name. Severity profiles reference
#: these names, so the set of names is a stable public interface.
OPS = {
    "swap_adjacent_chars": swap_adjacent_chars,
    "delete_char": delete_char,
    "duplicate_char": duplicate_char,
    "insert_random_char": insert_random_char,
    "nearby_key_substitution": nearby_key_substitution,
    "split_word": split_word,
    "merge_words": merge_words,
    "add_extra_space": add_extra_space,
    "flip_case": flip_case,
    "lowercase_sentence_start": lowercase_sentence_start,
    "random_capitalize_word": random_capitalize_word,
    "remove_punctuation": remove_punctuation,
    "duplicate_punctuation": duplicate_punctuation,
    "replace_punctuation": replace_punctuation,
    "delete_filler_word": delete_filler_word,
    "repeat_word": repeat_word,
    "swap_adjacent_words": swap_adjacent_words,
    "abbreviate_word": abbreviate_word,
    "drop_apostrophe": drop_apostrophe,
}
