#### Datei 3: `src/evaluation.py`

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer

E5_MODEL = "intfloat/multilingual-e5-base"

MODELS = {
    "v1": "intfloat/multilingual-e5-base",
    "v2": "BAAI/bge-m3",
    "v2-bad": "BAAI/bge-m3",
}

EXCLUDED_CHUNKS = {
    "v2-bad": {"docs/release-notes.md#v2-4-postgres"},
}


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    return float(bool(set(retrieved_ids[:k]) & relevant_ids))


def dcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    score = 0.0
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    ideal_hits = min(len(relevant_ids), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg_at_k(retrieved_ids, relevant_ids, k) / ideal if ideal else 0.0


class LocalVectorIndex:
    def __init__(self, records: List[Dict[str, Any]], vectors: np.ndarray):
        self.records = records
        self.vectors = vectors.astype(np.float32)

    def search(self, query_vector: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
        scores = self.vectors @ query_vector.astype(np.float32)
        positions = np.argsort(-scores)[:limit]
        return [{**self.records[idx], "score": float(scores[idx])} for idx in positions]


def encode_passages(model: SentenceTransformer, model_name: str, texts: List[str]) -> np.ndarray:
    if model_name == E5_MODEL:
        texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def encode_queries(model: SentenceTransformer, model_name: str, queries: List[str]) -> np.ndarray:
    if model_name == E5_MODEL:
        queries = [f"query: {q}" for q in queries]
    return model.encode(queries, normalize_embeddings=True, show_progress_bar=False)


def run_evaluation(
    version: str,
    documents: List[Dict[str, Any]],
    golden_queries: List[Dict[str, Any]],
    top_k: int = 5,
    model_cache: Optional[Dict[str, SentenceTransformer]] = None,
) -> Dict[str, Any]:
    model_name = MODELS[version]
    cache = model_cache if model_cache is not None else {}

    if model_name not in cache:
        cache[model_name] = SentenceTransformer(model_name)
    model = cache[model_name]

    excluded = EXCLUDED_CHUNKS.get(version, set())
    docs = [d for d in documents if d["id"] not in excluded]

    doc_vectors = encode_passages(model, model_name, [d["text"] for d in docs])
    index = LocalVectorIndex(docs, doc_vectors)

    query_texts = [q["query"] for q in golden_queries]
    query_vectors = encode_queries(model, model_name, query_texts)

    query_results = []
    for item, q_vec in zip(golden_queries, query_vectors):
        hits = index.search(q_vec, limit=top_k)
        retrieved_ids = [h["id"] for h in hits]
        relevant_ids = set(item["relevant_chunk_ids"])

        query_results.append({
            "id": item["id"],
            "query": item["query"],
            "segment": item["segment"],
            "critical": item["critical"],
            "relevant_chunk_ids": item["relevant_chunk_ids"],
            "retrieved_chunk_ids": retrieved_ids,
            "top_1_score": hits[0]["score"] if hits else 0.0,
            "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, top_k),
            "rr": reciprocal_rank(retrieved_ids, relevant_ids),
            "ndcg_at_k": ndcg_at_k(retrieved_ids, relevant_ids, top_k),
        })

    critical_rows = [r for r in query_results if r["critical"]]

    metrics = {
        f"recall_at_{top_k}": sum(r["recall_at_k"] for r in query_results) / len(query_results),
        f"mrr_at_{top_k}": sum(r["rr"] for r in query_results) / len(query_results),
        f"ndcg_at_{top_k}": sum(r["ndcg_at_k"] for r in query_results) / len(query_results),
        "critical_pass_rate": (
            sum(r["recall_at_k"] for r in critical_rows) / len(critical_rows)
            if critical_rows else 1.0
        ),
    }

    return {
        "version": version,
        "model_name": model_name,
        "doc_count": len(docs),
        "metrics": metrics,
        "queries": query_results,
    }


def main() -> None:
    data_dir = Path("data")
    documents = json.loads((data_dir / "corpus.json").read_text(encoding="utf-8"))
    golden_queries = json.loads((data_dir / "golden_queries.json").read_text(encoding="utf-8"))

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    cache: Dict[str, SentenceTransformer] = {}

    for version in ("v1", "v2", "v2-bad"):
        report = run_evaluation(version, documents, golden_queries, model_cache=cache)
        out_path = out_dir / f"report_{version.replace('-', '_')}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report gespeichert: {out_path} -> {report['metrics']}")


if __name__ == "__main__":
    main()
