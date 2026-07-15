"""Extract clean utterances from the DailyDialog parquet files.

Source: roskoN/dailydialog parquet mirror (original yanran.li host is dead).
Utterances are pre-tokenised; we detokenise them into natural sentences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from .detokenize import detokenize


def iter_utterances(parquet_path: str | Path) -> Iterator[str]:
    """Yield detokenised utterances, streaming batch by batch."""
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(columns=["utterances"]):
        for dialogue in batch.column("utterances").to_pylist():
            for utterance in dialogue:
                text = detokenize(utterance)
                if text:
                    yield text
