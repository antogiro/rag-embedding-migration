from __future__ import annotations

import json

from rag_embedding_migration import ARTIFACTS_DIR

def load_report(version: str) -> dict:
    with (ARTIFACTS_DIR / f"evaluation-{version}.json").open(encoding="utf-8") as file:
        return json.load(file)

def main() -> None:
    old = load_report("v1")
    new = load_report("v2")

    print(f"Alt: {old['embedding_model']}")
    print(f"Neu: {new['embedding_model']}\n")

    for metric in ("recall_at_5" , "mrr_at_5", "ndcg_at_5", "critical_pass_rate"):
        old_value = old["metrics"][metric]
        new_value = new["metrics"][metric]
        print(
            f"{metric:20} "
            f"{old_value:.3f} -> {new_value:.3f} "
            f"({new_value - old_value:+.3f})"
        )

    old_rows = {row["id"]: row for row in old["queries"]}
    new_rows = {row["id"]: row for row in new["queries"]}

    regressions = [
        (query_id, old_rows[query_id], new_row)
        for query_id, new_row in new_rows.items()
        if old_rows[query_id]["recall_at_k"] > new_row["recall_at_k"]
    ]

    if regressions:
        print("\nRetrieval-Regressionen:")
        for query_id, old_row, new_row in regressions:
            print(f"- {query_id}: {old_row['retrieved_chunk_ids']} -> {new_row['retrieved_chunk_ids']}")

    else:
        print("\nKeine Racall@5-Regressionen.")

if __name__ == "__main__":
    main()