"""Synthetic noise generation: turn clean sentences into realistic typo'd input."""

from .generator import PROFILES, CorruptionResult, NoiseGenerator, SeverityProfile
from .keyboard import QWERTY_ADJACENCY, nearby_keys
from .ops import OPS

__all__ = [
    "PROFILES",
    "CorruptionResult",
    "NoiseGenerator",
    "SeverityProfile",
    "QWERTY_ADJACENCY",
    "nearby_keys",
    "OPS",
]
