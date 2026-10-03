from __future__ import annotations

def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return float(bool(set(retrieved_ids[:k]) & relevant_ids))

def dcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    score = 0.0
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            score += 1.0 / __import__("math").log2(rank + 1)
    return score

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    ideal_hits = min(len(relevant_ids), k)
    ideal = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg_at_k(retrieved_ids, relevant_ids, k) / ideal if ideal else 0.0