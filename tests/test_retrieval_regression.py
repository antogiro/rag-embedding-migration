from rag_embedding_migration.evaluate import evaluate

MAX_RECALL_DROP = 0.02
MAX_MRR_DROP = 0.02

def test_candidate_embedding_model_not_regress() -> None:
    baseline = evaluate("v1")
    candidate = evaluate("v2")

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]

    assert (
        candidate_metrics["recall_at_5"]
        >= baseline_metrics["recall_at_5"] - MAX_RECALL_DROP
    ), (
        "Recall@5 ist gegenüber dem alten Index zu stark gefallen: "
        f"{baseline_metrics['recall_at_5']:.3f} ->"
        f"{candidate_metrics['recall_at_5']:.3f}"
    )

    assert (
        candidate_metrics["mrr_at_5"]
        >= baseline_metrics["mrr_at_5"] - MAX_MRR_DROP
    ), (
        "MRR@5 ist gegenüber dem alten Index zu stark gefallen: "
        f"{baseline_metrics['mrr_at_5']:.3f} -> "
        f"{candidate_metrics['mrr-at_5']:.3f}"
    )

def test_every_critical_query_has_a_relevant_hit() -> None:
    candidate = evaluate("v2")

    failed = [
        row["id"]
        for row in candidate["queries"]
        if row["critical"] and row["recall_at_k"] == 0
    ]

    assert not failed, f"Kritische Queries ohne relevanten Treffer in Top 5: {failed}"