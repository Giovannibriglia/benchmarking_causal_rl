from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, Optional


class CSVLogger:
    """Lightweight CSV logger; appends rows with consistent fieldnames."""

    def __init__(
        self, filepath: str, fieldnames: Optional[Iterable[str]] = None
    ) -> None:
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.fieldnames = list(fieldnames) if fieldnames is not None else None
        self._writer = None
        self._file = None
        self._file_had_data = False

    def __enter__(self):
        # Open in append mode; if empty, write header once
        file_exists = os.path.exists(self.filepath)
        self._file_had_data = file_exists and os.path.getsize(self.filepath) > 0
        self._file = open(self.filepath, "a", newline="")

        if self.fieldnames is None:
            raise ValueError("CSVLogger requires fieldnames for stable schema.")

        if not self._file_had_data:
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
        else:
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def log(self, row: Dict) -> None:
        if self._writer is None:
            raise RuntimeError("CSVLogger must be used within context manager.")
        filtered = {k: row.get(k, "") for k in self.fieldnames}
        self._writer.writerow(filtered)
        self._file.flush()
