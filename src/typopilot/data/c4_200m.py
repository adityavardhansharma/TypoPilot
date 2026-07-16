"""Stream (errorful -> clean) pairs from a C4_200M parquet shard.

Source: liweili/c4_200m on Hugging Face (materialised pairs from Google's
C4_200M synthetic GEC dataset; underlying C4 text is ODC-BY — commercial
use permitted with attribution).

Columns: ``input`` (corrupted sentence), ``output`` (clean sentence).
These corruptions are grammar-flavoured (tagged corruption model), which
complements our keyboard/misspelling synthetic noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


def iter_pairs(parquet_path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (errorful, clean) pairs, streaming batch by batch."""
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(columns=["input", "output"], batch_size=4096):
        inputs = batch.column("input").to_pylist()
        outputs = batch.column("output").to_pylist()
        for errorful, clean in zip(inputs, outputs):
            if errorful and clean:
                yield errorful, clean
