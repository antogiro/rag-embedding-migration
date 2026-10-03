from __future__ import annotations

import argparse
import json
from collections import defaultdict

from rag_embedding_migration.config import (
    ARTIFACTS_DIR,
    GOLDEN_QUERIES_PATH,
    MODELS,
    TOP_K
)
from rag_embedding_migration.corpus import load_jsonl
from rag_embedding_migration.embeddings import embed_queries, load_model
from rag_embedding_migration.index import LocalVectorIndex
from rag_embedding_migration.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

def evaluate(version: str, top_k: int = TOP_K) -> dict:
    queries = load_jsonl(GOLDEN_QUERIES_PATH)
    model = load_model(MODELS[version])
    index = LocalVectorIndex.load(ARTIFACTS_DIR / f"index-{version}")

    query_vectors = embed_queries(model, [item["query"] for item in queries])

    rows = []
    for item, vector in zip(queries, query_vectors, strict=True):
        results = index.search(vector, limit=top_k)
        retrieved_ids = [result["id"] for result in results]
        relevant_ids = set(item["relevant_chunk_ids"])

        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "segment": item["segment"],
                "critical": item["critical"],
                "relevant_chunk_ids": item["relevant_chunk_ids"],
                "retrieved_chunk_ids": retrieved_ids,
                "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, top_k),
                "rr": reciprocal_rank(retrieved_ids, relevant_ids),
                "ndcg_at_k": ndcg_at_k(retrieved_ids, relevant_ids, top_k),
            }
        )

    metrics = {
        f"recall_at_{top_k}": sum(row["recall_at_k"] for row in rows) / len(rows),
        f"mrr_at_{top_k}": sum(row["rr"] for row in rows) / len(rows),
        f"ndcg_at_{top_k}": sum(row["ndcg_at_k"] for row in rows) / len(rows),
        "critical_pass_rate": (
            sum(
                row["recall_at_k"]
                for row in rows
                if row["critical"]
            )
            / sum(1 for row in rows if row["critical"])
        ),
    }

    by_segment = defaultdict(list)
    for row in rows:
        by_segment[row["segement"]].append(row)

    segment_metrics = {
        segment: {
            f"recall_at_{top_k}": sum(row["recall_at_k"] for row in values) / len(values),
            f"mrr_at_{top_k}": sum(row["rr"] for row in values) / len(values),
        }
        for segment, values in by_segment.items()
    }

    report = {
        "index_version": version,
        "embedding_model": MODELS[version],
        "top_k": top_k,
        "metrics": metrics,
        "segment_metrics": segment_metrics,
        "queries": rows,
    }

    output_path = ARTIFACTS_DIR / f"evaluation-{version}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(MODELS))
    parser.add_argument("--all", action="store_true")

    if not args.version and not args.all:
        parser.error("Bitte --version v1|v2 oder -all angeben.")

    versions = sorted(MODELS) if args.all else [args.version]

    for version in versions:
        report = evaluate(version)
        print(f"\n{version}: {report['embedding_model']}")
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))

    if __name__ == "__main__":
        main()