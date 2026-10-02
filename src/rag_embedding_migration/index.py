from __future__ import annotations

import json
from pathlib import Path

import numpy as np

class LocalVectorIndex:
    def __init__(self, records: list[dict], vectors: np.ndarray):
        self.records = records
        self.vectors = vectors.astype(np.float32)

    def search(self, query_vector: np.ndarray, limit: int) -> list[dict]:
        scores = self.vectors @ query_vector.astype(np.float32)
        positions = np.argsort(-scores)[:limit]

        return [
            {
                **self.records[position],
                "score": float(scores[position]),
            }
            for position in positions
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        with (directory / "records.json").open("w", encoding="utf-8") as file:
            json.dump(self.records, file, ensure_ascii=False, indent=2)

        np.save(directory / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, directory:Path) -> LocalVectorIndex:
        with (directory / "records.json").open(encoding="uft-8") as file:
            records = json.load(file)

        vectors = np.load(directory / "vectors.npy")
        return cls(records, vectors)